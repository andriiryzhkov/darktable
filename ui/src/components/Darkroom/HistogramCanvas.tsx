import { useRef, useEffect } from "react";
import type { HistogramData } from "./histogramCompute";

interface Props {
  data: HistogramData | null;
  width: number;
  height: number;
  logarithmic?: boolean;
  showR?: boolean;
  showG?: boolean;
  showB?: boolean;
}

export default function HistogramCanvas({ data, width, height, logarithmic = true, showR = true, showG = true, showB = true }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    if (!data || data.max === 0) return;

    const bins = 256;
    const binW = width / bins;
    const logMax = logarithmic ? Math.log1p(data.max) : data.max;

    const drawChannel = (bins_data: Uint32Array, color: string) => {
      ctx.beginPath();
      ctx.moveTo(0, height);

      for (let i = 0; i < bins; i++) {
        const x = i * binW;
        const raw = bins_data[i];
        const v = raw > 0
          ? (logarithmic ? Math.log1p(raw) / logMax : raw / logMax)
          : 0;
        const y = height - v * height;
        ctx.lineTo(x, y);
      }

      ctx.lineTo(width, height);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
    };

    // Draw grid first (behind channels): 4 divisions = 3 lines each axis
    const gridColor = "rgba(255, 255, 255, 0.1)";
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    for (let k = 1; k < 4; k++) {
      const x = Math.round(width * k / 4) + 0.5;
      const y = Math.round(height * k / 4) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }

    ctx.globalCompositeOperation = "screen";
    if (showR) drawChannel(data.r, "rgba(200, 0, 0, 0.8)");
    if (showG) drawChannel(data.g, "rgba(0, 200, 0, 0.8)");
    if (showB) drawChannel(data.b, "rgba(0, 0, 200, 0.8)");
    ctx.globalCompositeOperation = "source-over";
  }, [data, width, height, logarithmic, showR, showG, showB]);

  return (
    <canvas
      ref={canvasRef}
      className="scope-canvas"
      style={{ width, height }}
    />
  );
}
