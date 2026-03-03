import { useCallback, useRef, useState } from "react";
import { useDevelopStore } from "../../stores/developStore";

export default function ExposureModule() {
  const { setExposure, setBlack } = useDevelopStore();
  const [exposureValue, setExposureValue] = useState(0);
  const [blackValue, setBlackValue] = useState(0);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleExposureChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = parseFloat(e.target.value);
      setExposureValue(value);

      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        setExposure(value);
      }, 150);
    },
    [setExposure],
  );

  const handleBlackChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = parseFloat(e.target.value);
      setBlackValue(value);

      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        setBlack(value);
      }, 150);
    },
    [setBlack],
  );

  return (
    <div className="p-4 border-b border-[var(--border-color)]">
      <h3 className="text-sm font-medium mb-3">Exposure</h3>

      <div className="space-y-3">
        <div>
          <div className="flex justify-between text-xs text-[var(--plugin-label-color)] mb-1">
            <span>Exposure</span>
            <span>{exposureValue.toFixed(2)} EV</span>
          </div>
          <input
            type="range"
            min="-4"
            max="4"
            step="0.01"
            value={exposureValue}
            onChange={handleExposureChange}
            className="w-full accent-[var(--bauhaus-fill)]"
          />
        </div>

        <div>
          <div className="flex justify-between text-xs text-[var(--plugin-label-color)] mb-1">
            <span>Black</span>
            <span>{blackValue.toFixed(4)}</span>
          </div>
          <input
            type="range"
            min="-0.1"
            max="0.1"
            step="0.001"
            value={blackValue}
            onChange={handleBlackChange}
            className="w-full accent-[var(--bauhaus-fill)]"
          />
        </div>
      </div>
    </div>
  );
}
