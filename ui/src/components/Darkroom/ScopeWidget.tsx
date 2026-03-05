import { useState, useMemo, useRef, useCallback, useEffect } from "react";
import { BarChart3, AudioWaveform, ChevronUp, ChevronRight } from "lucide-react";

const S = 12;
const LinearIcon = () => (
  <svg width={S} height={S} viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round">
    <line x1="2" y1="10" x2="10" y2="2" />
  </svg>
);
const LogIcon = () => (
  <svg width={S} height={S} viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round">
    <path d="M2 10 Q2 2, 10 2" />
  </svg>
);
import { useDevelopStore } from "../../stores/developStore";
import { computeHistogram, computeWaveform } from "./histogramCompute";
import HistogramCanvas from "./HistogramCanvas";
import WaveformCanvas from "./WaveformCanvas";

type ScopeMode = "histogram" | "waveform";
type ScaleMode = "logarithmic" | "linear";
type WaveOrientation = "vertical" | "horizontal";

/**
 * Decode a JPEG data URL to RGBA Uint8Array via offscreen canvas.
 */
function decodeJpegToRgba(
  src: string,
): Promise<Uint8Array | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) { resolve(null); return; }
      ctx.drawImage(img, 0, 0);
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      resolve(new Uint8Array(imgData.data.buffer));
    };
    img.onerror = () => resolve(null);
    img.src = src;
  });
}

export default function ScopeWidget() {
  const [mode, setMode] = useState<ScopeMode>("histogram");
  const [scale, setScale] = useState<ScaleMode>("logarithmic");
  const [waveOrientation, setWaveOrientation] = useState<WaveOrientation>("vertical");
  const [showR, setShowR] = useState(true);
  const [showG, setShowG] = useState(true);
  const [showB, setShowB] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 300, h: 150 });

  const frameData = useDevelopStore((s) => s.frameData);
  const previewSrc = useDevelopStore((s) => s.previewSrc);
  const previewWidth = useDevelopStore((s) => s.previewWidth);
  const previewHeight = useDevelopStore((s) => s.previewHeight);

  // When only JPEG is available, decode to RGBA pixels
  const [jpegPixels, setJpegPixels] = useState<Uint8Array | null>(null);
  const [jpegSize, setJpegSize] = useState({ w: 0, h: 0 });

  useEffect(() => {
    if (frameData || !previewSrc) {
      setJpegPixels(null);
      return;
    }
    let cancelled = false;
    decodeJpegToRgba(previewSrc).then((data) => {
      if (!cancelled && data) {
        setJpegPixels(data);
        setJpegSize({ w: previewWidth, h: previewHeight });
      }
    });
    return () => { cancelled = true; };
  }, [frameData, previewSrc, previewWidth, previewHeight]);

  const pixelSource = frameData ?? jpegPixels;
  const isBGRA = !!frameData;
  const pixelWidth = frameData ? previewWidth : jpegSize.w;
  const pixelHeight = frameData ? previewHeight : jpegSize.h;
  const hasImage = !!pixelSource;

  // Track container width
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0].contentRect.width;
      if (w > 0) setSize({ w: Math.round(w), h: Math.min(150, Math.round(w * 0.7)) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const histData = useMemo(() => {
    if (!pixelSource) return null;
    return computeHistogram(pixelSource, isBGRA);
  }, [pixelSource, isBGRA]);

  const waveData = useMemo(() => {
    if (!pixelSource || mode !== "waveform") return null;
    return computeWaveform(pixelSource, pixelWidth, pixelHeight, isBGRA);
  }, [pixelSource, pixelWidth, pixelHeight, isBGRA, mode]);

  const toggleScale = useCallback(() => {
    setScale((s) => (s === "logarithmic" ? "linear" : "logarithmic"));
  }, []);

  const isLog = scale === "logarithmic";

  return (
    <div className="scope-widget" ref={containerRef}>
      <div className="scope-display">
        {mode === "histogram" ? (
          <HistogramCanvas data={histData} width={size.w} height={size.h} logarithmic={isLog} showR={showR} showG={showG} showB={showB} />
        ) : (
          <WaveformCanvas data={waveData} width={size.w} height={size.h} horizontal={waveOrientation === "horizontal"} showR={showR} showG={showG} showB={showB} />
        )}
        {!hasImage && (
          <span className="scope-placeholder">no image</span>
        )}

        {/* Overlay toolbar — visible on hover */}
        <div className="scope-toolbar">
          <div className="scope-toolbar-left">
            <button
              className="scope-btn"
              data-active={mode === "histogram"}
              title="histogram"
              onClick={() => setMode("histogram")}
            >
              <BarChart3 size={12} />
            </button>
            <button
              className="scope-btn"
              data-active={mode === "waveform"}
              title="waveform"
              onClick={() => setMode("waveform")}
            >
              <AudioWaveform size={12} />
            </button>
          </div>

          <div className="scope-toolbar-right">
            {mode === "histogram" ? (
              <button
                className="scope-btn"
                data-active={isLog}
                title={isLog ? "logarithmic scale" : "linear scale"}
                onClick={toggleScale}
              >
                {isLog ? <LogIcon /> : <LinearIcon />}
              </button>
            ) : (
              <button
                className="scope-btn"
                title={waveOrientation === "vertical" ? "vertical" : "horizontal"}
                onClick={() => setWaveOrientation((o) => o === "vertical" ? "horizontal" : "vertical")}
              >
                {waveOrientation === "horizontal" ? <ChevronUp size={12} /> : <ChevronRight size={12} />}
              </button>
            )}
            <button className="scope-btn scope-ch-r" data-active={showR} title="red" onClick={() => setShowR((v) => !v)} />
            <button className="scope-btn scope-ch-g" data-active={showG} title="green" onClick={() => setShowG((v) => !v)} />
            <button className="scope-btn scope-ch-b" data-active={showB} title="blue" onClick={() => setShowB((v) => !v)} />
          </div>
        </div>
      </div>
    </div>
  );
}
