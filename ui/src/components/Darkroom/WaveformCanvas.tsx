import { useRef, useEffect } from "react";
import type { WaveformData } from "./histogramCompute";

interface Props {
  data: WaveformData | null;
  width: number;
  height: number;
  horizontal?: boolean;
  showR?: boolean;
  showG?: boolean;
  showB?: boolean;
}

export default function WaveformCanvas({ data, width, height, horizontal = false, showR = true, showG = true, showB = true }: Props) {
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

    const { r, g, b, cols, rows } = data;

    // Render into offscreen canvas at data resolution
    // Vertical: X=image column, Y=brightness (default)
    // Horizontal: X=brightness, Y=image column (rotated 90° CW)
    const offW = horizontal ? rows : cols;
    const offH = horizontal ? cols : rows;
    const off = document.createElement("canvas");
    off.width = offW;
    off.height = offH;
    const offCtx = off.getContext("2d")!;
    const imgData = offCtx.createImageData(offW, offH);
    const pixels = imgData.data;

    for (let col = 0; col < cols; col++) {
      const base = col * rows;
      for (let row = 0; row < rows; row++) {
        const rv = showR ? r[base + row] : 0;
        const gv = showG ? g[base + row] : 0;
        const bv = showB ? b[base + row] : 0;

        if (rv === 0 && gv === 0 && bv === 0) continue;

        let px: number, py: number;
        if (horizontal) {
          // X = brightness (row), Y = image column (flipped so left→top)
          px = row;
          py = col;
        } else {
          // X = image column, Y = brightness (row 0=dark at bottom)
          px = col;
          py = rows - 1 - row;
        }

        const pi = (py * offW + px) * 4;
        pixels[pi] = Math.min(255, pixels[pi] + rv);
        pixels[pi + 1] = Math.min(255, pixels[pi + 1] + gv);
        pixels[pi + 2] = Math.min(255, pixels[pi + 2] + bv);
        pixels[pi + 3] = 255;
      }
    }

    offCtx.putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, width, height);

    // Draw waveform gridlines: 9 divisions = 8 lines
    // Lines 1 and 5 are dashed reference lines (thicker)
    const gridColor = "rgba(255, 255, 255, 0.1)";
    ctx.strokeStyle = gridColor;
    for (let k = 1; k < 9; k++) {
      const isDashed = k === 1 || k === 5;
      ctx.lineWidth = isDashed ? (k === 1 ? 1.5 : 1.5) : 1;
      ctx.setLineDash(isDashed ? [4, 4] : []);
      const pos = Math.round((horizontal ? width : height) * k / 9) + 0.5;
      ctx.beginPath();
      if (horizontal) {
        ctx.moveTo(pos, 0); ctx.lineTo(pos, height);
      } else {
        ctx.moveTo(0, pos); ctx.lineTo(width, pos);
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }, [data, width, height, horizontal, showR, showG, showB]);

  return (
    <canvas
      ref={canvasRef}
      className="scope-canvas"
      style={{ width, height }}
    />
  );
}
