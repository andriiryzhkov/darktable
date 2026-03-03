interface ModuleSliderProps {
  min: number;
  max: number;
  value: number;
  step?: number;
  onChange?: (value: number) => void;
  displayValue?: string;
}

export default function ModuleSlider({
  min,
  max,
  value,
  step = 1,
  onChange,
  displayValue,
}: ModuleSliderProps) {
  return (
    <div className="flex items-center gap-2">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange?.(Number(e.target.value))}
        className="dt-slider flex-1"
      />
      {displayValue !== undefined && (
        <span
          className="text-xs shrink-0"
          style={{ color: "var(--fg-color)", width: 24, textAlign: "right" }}
        >
          {displayValue}
        </span>
      )}
    </div>
  );
}
