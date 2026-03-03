import { useEffect, useRef } from "react";
import { useDevelopStore } from "../../stores/developStore";

export default function PreviewCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { frameData, previewWidth, previewHeight } = useDevelopStore();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frameData || frameData.length === 0) return;

    canvas.width = previewWidth;
    canvas.height = previewHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(previewWidth, previewHeight);
    const dst = imageData.data;

    // Convert BGRA (from server) to RGBA (canvas expects RGBA)
    for (let i = 0; i < frameData.length; i += 4) {
      dst[i] = frameData[i + 2];     // R <- B
      dst[i + 1] = frameData[i + 1]; // G <- G
      dst[i + 2] = frameData[i];     // B <- R
      dst[i + 3] = frameData[i + 3]; // A <- A
    }

    ctx.putImageData(imageData, 0, 0);
  }, [frameData, previewWidth, previewHeight]);

  if (!frameData) {
    return (
      <div className="text-[var(--plugin-label-color)]">
        Rendering preview...
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      className="max-w-full max-h-full object-contain"
    />
  );
}
