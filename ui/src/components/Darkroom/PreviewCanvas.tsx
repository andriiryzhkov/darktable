import { useEffect, useRef } from "react";
import { useDevelopStore } from "../../stores/developStore";

export default function PreviewCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { previewSrc, frameData, previewWidth, previewHeight, previewError, loading } = useDevelopStore();

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

  // JPEG path: browser-native decode via <img> (hardware accelerated)
  if (previewSrc) {
    return (
      <img
        src={previewSrc}
        width={previewWidth}
        height={previewHeight}
        alt="preview"
        draggable={false}
        className="max-w-full max-h-full object-contain"
        onError={() => console.error("[preview] img decode failed, src length:", previewSrc.length)}
      />
    );
  }

  // Raw BGRA fallback path
  return (
    <canvas
      ref={canvasRef}
      className="max-w-full max-h-full object-contain"
    />
  );
}
