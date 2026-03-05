import { useCallback, useRef, type ReactNode } from "react";

interface BauhausSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  defaultValue?: number;
  origin?: number;
  gradient?: string;
  color?: string;
  format?: (v: number) => string;
  onChange?: (value: number) => void;
  actionIcon?: ReactNode;
  onAction?: () => void;
}

function defaultFormat(v: number): string {
  if (Number.isInteger(v)) return String(v);
  return v.toPrecision(4);
}

export default function BauhausSlider({
  label,
  value,
  min,
  max,
  step = 0.01,
  defaultValue,
  origin,
  gradient,
  color,
  format = defaultFormat,
  onChange,
  actionIcon,
  onAction,
}: BauhausSliderProps) {
  const trackRef = useRef<HTMLDivElement>(null);

  const valueFromX = useCallback(
    (clientX: number) => {
      const track = trackRef.current;
      if (!track) return value;
      const rect = track.getBoundingClientRect();
      const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      const raw = min + pct * (max - min);
      return Math.round(raw / step) * step;
    },
    [min, max, step, value],
  );

  const onDoubleClick = useCallback(() => {
    if (defaultValue !== undefined) {
      onChange?.(defaultValue);
    }
  }, [defaultValue, onChange]);

  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      onChange?.(valueFromX(e.clientX));
    },
    [onChange, valueFromX],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!(e.currentTarget as HTMLElement).hasPointerCapture(e.pointerId))
        return;
      onChange?.(valueFromX(e.clientX));
    },
    [onChange, valueFromX],
  );

  const onPointerUp = useCallback((e: React.PointerEvent) => {
    const el = e.currentTarget as HTMLElement;
    if (el.hasPointerCapture(e.pointerId)) {
      el.releasePointerCapture(e.pointerId);
    }
  }, []);

  const range = max - min;
  const orig = origin ?? min;
  const originPct = ((orig - min) / range) * 100;
  const valuePct = ((value - min) / range) * 100;
  const fillLeft = Math.min(originPct, valuePct);
  const fillWidth = Math.abs(valuePct - originPct);

  return (
    <div className="bauhaus-slider">
      <div
        className="bauhaus-slider-body"
        onDoubleClick={onDoubleClick}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        <div className="bauhaus-slider-header">
          <span className="bauhaus-slider-label">{label}</span>
          <span className="bauhaus-slider-value">{format(value)}</span>
        </div>
        <div
          ref={trackRef}
          className="bauhaus-slider-track"
          style={gradient ? { background: gradient } : undefined}
        >
          {color && (
            <div
              className="bauhaus-slider-color"
              style={{ backgroundColor: color }}
            />
          )}
          <div
            className="bauhaus-slider-fill"
            style={{ left: `${fillLeft}%`, width: `${fillWidth}%` }}
          />
          <div
            className="bauhaus-slider-indicator"
            style={{ left: `${valuePct}%` }}
          />
        </div>
      </div>
      <div className="bauhaus-slider-action" onClick={onAction}>
        {actionIcon}
      </div>
    </div>
  );
}
