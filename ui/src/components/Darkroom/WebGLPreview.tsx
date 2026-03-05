import { useEffect, useRef, useCallback } from "react";
import { useDevelopStore, ZOOM_LEVELS } from "../../stores/developStore";

const VERTEX_SRC = `#version 300 es
in vec2 a_position;
out vec2 v_texcoord;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  v_texcoord = a_position * 0.5 + 0.5;
  v_texcoord.y = 1.0 - v_texcoord.y;
}`;

const FRAGMENT_SRC = `#version 300 es
precision mediump float;
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
void main() {
  vec4 c = texture(u_texture, v_texcoord);
  fragColor = vec4(c.b, c.g, c.r, 1.0);
}`;

function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("[WebGLPreview] shader compile:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function createProgram(gl: WebGL2RenderingContext): WebGLProgram | null {
  const vs = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SRC);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SRC);
  if (!vs || !fs) return null;

  const prog = gl.createProgram();
  if (!prog) return null;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error("[WebGLPreview] program link:", gl.getProgramInfoLog(prog));
    gl.deleteProgram(prog);
    return null;
  }
  // Shaders are linked — safe to delete references
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return prog;
}

export default function WebGLPreview() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const textureRef = useRef<WebGLTexture | null>(null);
  const vaoRef = useRef<WebGLVertexArrayObject | null>(null);

  const frameData = useDevelopStore((s) => s.frameData);
  const previewWidth = useDevelopStore((s) => s.previewWidth);
  const previewHeight = useDevelopStore((s) => s.previewHeight);
  const zoom = useDevelopStore((s) => s.zoom);
  const panX = useDevelopStore((s) => s.panX);
  const panY = useDevelopStore((s) => s.panY);
  const setPan = useDevelopStore((s) => s.setPan);
  const setZoom = useDevelopStore((s) => s.setZoom);

  // Initialize WebGL on mount
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
    if (!gl) {
      console.error("[WebGLPreview] WebGL 2 not supported");
      return;
    }
    glRef.current = gl;

    const prog = createProgram(gl);
    if (!prog) return;
    programRef.current = prog;

    // Create fullscreen quad VAO
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    vaoRef.current = vao;

    const posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    // Two triangles forming a fullscreen quad
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1,  -1, 1,
      -1,  1,  1, -1,   1, 1,
    ]), gl.STATIC_DRAW);

    const aPos = gl.getAttribLocation(prog, "a_position");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    gl.bindVertexArray(null);

    // Create texture
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    textureRef.current = tex;

    return () => {
      gl.deleteTexture(tex);
      gl.deleteProgram(prog);
      gl.deleteVertexArray(vao);
      gl.deleteBuffer(posBuffer);
      glRef.current = null;
      programRef.current = null;
      textureRef.current = null;
      vaoRef.current = null;
    };
  }, []);

  // Upload pixels and draw when frameData changes
  useEffect(() => {
    const gl = glRef.current;
    const prog = programRef.current;
    const tex = textureRef.current;
    const vao = vaoRef.current;
    const canvas = canvasRef.current;
    if (!gl || !prog || !tex || !vao || !canvas || !frameData || frameData.length === 0) return;

    // Resize canvas to match preview dimensions
    if (canvas.width !== previewWidth || canvas.height !== previewHeight) {
      canvas.width = previewWidth;
      canvas.height = previewHeight;
    }

    gl.viewport(0, 0, previewWidth, previewHeight);

    // Upload BGRA pixels as RGBA (shader swaps B↔R)
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA,
      previewWidth, previewHeight, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, frameData,
    );

    // Draw
    gl.useProgram(prog);
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
  }, [frameData, previewWidth, previewHeight]);

  // Drag to pan
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ startX: number; startY: number; startPanX: number; startPanY: number } | null>(null);
  const isZoomed = zoom !== "fit" && zoom !== "fill" && zoom !== "small";

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      if (e.button === 1 || (e.button === 0 && isZoomed)) {
        e.preventDefault();
        e.currentTarget.setPointerCapture(e.pointerId);
        dragRef.current = { startX: e.clientX, startY: e.clientY, startPanX: panX, startPanY: panY };
      }
    },
    [isZoomed, panX, panY],
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragRef.current || !isZoomed) return;
      const container = containerRef.current;
      if (!container) return;

      const zoomFactor = parseInt(zoom) / 100;
      const rect = container.getBoundingClientRect();
      const dx = (e.clientX - dragRef.current.startX) / (rect.width * zoomFactor);
      const dy = (e.clientY - dragRef.current.startY) / (rect.height * zoomFactor);
      setPan(
        Math.max(0, Math.min(1, dragRef.current.startPanX - dx)),
        Math.max(0, Math.min(1, dragRef.current.startPanY - dy)),
      );
    },
    [zoom, isZoomed, setPan],
  );

  const handlePointerUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const idx = ZOOM_LEVELS.indexOf(zoom);
      if (e.deltaY < 0 && idx < ZOOM_LEVELS.length - 1) {
        setZoom(ZOOM_LEVELS[idx + 1]);
      } else if (e.deltaY > 0 && idx > 0) {
        setZoom(ZOOM_LEVELS[idx - 1]);
      }
    },
    [zoom, setZoom],
  );

  const zoomFactor = isZoomed ? parseInt(zoom) / 100 : 1;
  const transformStyle = isZoomed
    ? {
        transform: `scale(${zoomFactor})`,
        transformOrigin: `${panX * 100}% ${panY * 100}%`,
      }
    : undefined;

  return (
    <div
      ref={containerRef}
      className="preview-container"
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onWheel={handleWheel}
      style={{ cursor: isZoomed ? "grab" : "default" }}
    >
      <canvas
        ref={canvasRef}
        className="max-w-full max-h-full object-contain"
        style={transformStyle}
      />
    </div>
  );
}
