import { useRef, useCallback, useEffect } from "react";
import { useDevelopStore, ZOOM_LEVELS, ZOOM_LABELS, isZoomedIn, getZoomFactor, getZoomLabel, type ZoomPreset } from "../../../stores/developStore";
import BauhausCombo from "../../controls/BauhausCombo";

const ZOOM_OPTIONS = ZOOM_LEVELS.map((l) => ZOOM_LABELS[l]);
const LABEL_TO_ZOOM = Object.fromEntries(
  ZOOM_LEVELS.map((l) => [ZOOM_LABELS[l], l]),
) as Record<string, ZoomPreset>;

const NAV_MAX_HEIGHT = 150;

export default function NavigationModule() {
  const previewSrc = useDevelopStore((s) => s.previewSrc);
  const frameData = useDevelopStore((s) => s.frameData);
  const previewWidth = useDevelopStore((s) => s.previewWidth);
  const previewHeight = useDevelopStore((s) => s.previewHeight);
  const zoom = useDevelopStore((s) => s.zoom);
  const panX = useDevelopStore((s) => s.panX);
  const panY = useDevelopStore((s) => s.panY);
  const setZoom = useDevelopStore((s) => s.setZoom);
  const setPan = useDevelopStore((s) => s.setPan);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Render raw BGRA frameData onto canvas when previewSrc is not available
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frameData || previewSrc || previewWidth <= 0 || previewHeight <= 0) return;

    canvas.width = previewWidth;
    canvas.height = previewHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(previewWidth, previewHeight);
    const dst = imageData.data;
    const src = frameData;
    // BGRA → RGBA swap
    for (let i = 0, len = previewWidth * previewHeight * 4; i < len; i += 4) {
      dst[i] = src[i + 2];     // R ← B
      dst[i + 1] = src[i + 1]; // G ← G
      dst[i + 2] = src[i];     // B ← R
      dst[i + 3] = 255;        // A
    }
    ctx.putImageData(imageData, 0, 0);
  }, [frameData, previewSrc, previewWidth, previewHeight]);

  // Whether we show the viewport rectangle
  const showRect = isZoomedIn(zoom);
  const zoomNumeric = showRect ? getZoomFactor(zoom) : 1;
  const viewW = Math.min(1, 1 / zoomNumeric);
  const viewH = Math.min(1, 1 / zoomNumeric);

  // Click/drag on the thumbnail to pan
  const updatePan = useCallback(
    (clientX: number, clientY: number) => {
      const el = containerRef.current;
      if (!el) return;
      const img = el.querySelector(".nav-module-image") as HTMLElement | null;
      if (!img) return;
      const rect = img.getBoundingClientRect();
      const x = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      const y = Math.max(0, Math.min(1, (clientY - rect.top) / rect.height));
      setPan(x, y);
    },
    [setPan],
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      if (!showRect) return;
      e.currentTarget.setPointerCapture(e.pointerId);
      updatePan(e.clientX, e.clientY);
    },
    [showRect, updatePan],
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!showRect || e.buttons === 0) return;
      updatePan(e.clientX, e.clientY);
    },
    [showRect, updatePan],
  );

  const handleZoomChange = useCallback(
    (label: string) => {
      const preset = LABEL_TO_ZOOM[label];
      if (!preset) return;
      // Named modes stay as strings; numeric presets become numbers
      const num = parseInt(preset);
      setZoom(isNaN(num) ? preset : num);
    },
    [setZoom],
  );

  // Viewport rect position (clamped)
  const rectLeft = Math.max(0, Math.min(1 - viewW, panX - viewW / 2));
  const rectTop = Math.max(0, Math.min(1 - viewH, panY - viewH / 2));

  const hasPreview = previewSrc || frameData;

  return (
    <div className="nav-module">
      <div
        ref={containerRef}
        className="nav-module-preview"
        style={showRect ? { cursor: "grab" } : undefined}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
      >
        {previewSrc ? (
          <img
            src={previewSrc}
            alt="navigation"
            draggable={false}
            className="nav-module-image"
            style={{
              maxHeight: NAV_MAX_HEIGHT,
              aspectRatio: `${previewWidth} / ${previewHeight}`,
            }}
          />
        ) : (
          <canvas
            ref={canvasRef}
            className="nav-module-image"
            style={{
              maxHeight: NAV_MAX_HEIGHT,
              aspectRatio: `${previewWidth} / ${previewHeight}`,
              display: frameData ? "block" : "none",
            }}
          />
        )}
        {!hasPreview && (
          <div
            className="nav-module-placeholder"
            style={{
              maxHeight: NAV_MAX_HEIGHT,
              aspectRatio: `${previewWidth} / ${previewHeight}`,
            }}
          />
        )}
        {showRect && (
          <>
            {/* Darken area outside viewport */}
            <div className="nav-module-overlay" style={{ top: 0, left: 0, right: 0, height: `${rectTop * 100}%` }} />
            <div className="nav-module-overlay" style={{ top: `${(rectTop + viewH) * 100}%`, left: 0, right: 0, bottom: 0 }} />
            <div className="nav-module-overlay" style={{ top: `${rectTop * 100}%`, left: 0, width: `${rectLeft * 100}%`, height: `${viewH * 100}%` }} />
            <div className="nav-module-overlay" style={{ top: `${rectTop * 100}%`, right: 0, width: `${(1 - rectLeft - viewW) * 100}%`, height: `${viewH * 100}%` }} />
            {/* Viewport border */}
            <div
              className="nav-module-rect"
              style={{
                left: `${rectLeft * 100}%`,
                top: `${rectTop * 100}%`,
                width: `${viewW * 100}%`,
                height: `${viewH * 100}%`,
              }}
            />
          </>
        )}
        <div className="nav-module-zoom" onPointerDown={(e) => e.stopPropagation()}>
          <BauhausCombo
            hideLabel
            options={[...ZOOM_OPTIONS]}
            value={getZoomLabel(zoom)}
            onChange={handleZoomChange}
          />
        </div>
      </div>
    </div>
  );
}
