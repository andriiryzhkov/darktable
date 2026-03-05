import { useEffect, useRef, useCallback } from "react";
import { useDevelopStore, ZOOM_LEVELS } from "../../stores/developStore";
import WebGLPreview from "./WebGLPreview";
import PickerOverlay from "./PickerOverlay";

export default function PreviewCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const { previewSrc, frameData, previewWidth, previewHeight, previewError, loading } = useDevelopStore();
  const zoom = useDevelopStore((s) => s.zoom);
  const panX = useDevelopStore((s) => s.panX);
  const panY = useDevelopStore((s) => s.panY);
  const setPan = useDevelopStore((s) => s.setPan);
  const setZoom = useDevelopStore((s) => s.setZoom);

  // Raw BGRA fallback path (only used when format === "raw")
  useEffect(() => {
    if (previewSrc || !frameData || frameData.length === 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = previewWidth;
    canvas.height = previewHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(previewWidth, previewHeight);
    const dst = imageData.data;

    for (let i = 0; i < frameData.length; i += 4) {
      dst[i] = frameData[i + 2];
      dst[i + 1] = frameData[i + 1];
      dst[i + 2] = frameData[i];
      dst[i + 3] = frameData[i + 3];
    }

    ctx.putImageData(imageData, 0, 0);
  }, [previewSrc, frameData, previewWidth, previewHeight]);

  // Drag to pan
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

  // Scroll wheel to zoom
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

  // Compute transform for zoom/pan
  const zoomFactor = isZoomed ? parseInt(zoom) / 100 : 1;

  const transformStyle = isZoomed
    ? {
        transform: `scale(${zoomFactor})`,
        transformOrigin: `${panX * 100}% ${panY * 100}%`,
      }
    : undefined;

  if (!previewSrc && !frameData) {
    return (
      <div className="text-[var(--plugin-label-color)] text-center">
        {previewError ? (
          <div>
            <div className="text-red-400 text-sm mb-1">Preview failed</div>
            <div className="text-xs opacity-70">{previewError}</div>
          </div>
        ) : (
          loading ? "Processing..." : "Rendering preview..."
        )}
      </div>
    );
  }

  // WebGL path for raw BGRA pixels — no JPEG encode/decode overhead
  if (frameData && frameData.length > 0) {
    return <WebGLPreview />;
  }

  const content = previewSrc ? (
    <img
      ref={imgRef}
      src={previewSrc}
      width={previewWidth}
      height={previewHeight}
      alt="preview"
      draggable={false}
      className="max-w-full max-h-full object-contain"
      style={transformStyle}
      onError={() => console.error("[preview] img decode failed, src length:", previewSrc.length)}
    />
  ) : (
    <canvas
      ref={canvasRef}
      className="max-w-full max-h-full object-contain"
      style={transformStyle}
    />
  );

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
      {content}
      <PickerOverlay targetRef={previewSrc ? imgRef : canvasRef} />
    </div>
  );
}
