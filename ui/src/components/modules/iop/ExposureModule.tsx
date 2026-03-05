import { useState, useEffect, useRef } from "react";
import { useDevelopStore } from "../../../stores/developStore";
import BauhausSlider from "../../controls/BauhausSlider";
import BauhausCheckbox from "../../controls/BauhausCheckbox";
import BauhausCombo from "../../controls/BauhausCombo";

export default function ExposureModule() {
  const exposureParams = useDevelopStore((s) => s.exposureParams);
  const setModuleParam = useDevelopStore((s) => s.setModuleParam);
  const fetchModuleParams = useDevelopStore((s) => s.fetchModuleParams);

  const [localExposure, setLocalExposure] = useState(0);
  const [localBlack, setLocalBlack] = useState(0);
  const [localPercentile, setLocalPercentile] = useState(50);
  const [localTarget, setLocalTarget] = useState(-4);
  const dragging = useRef(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Fetch params on mount if not already loaded
  useEffect(() => {
    if (!exposureParams) {
      fetchModuleParams("exposure");
    }
  }, [exposureParams, fetchModuleParams]);

  // Sync local slider state from store when not dragging
  useEffect(() => {
    if (exposureParams && !dragging.current) {
      setLocalExposure(exposureParams.exposure);
      setLocalBlack(exposureParams.black);
      setLocalPercentile(exposureParams.deflicker_percentile);
      setLocalTarget(exposureParams.deflicker_target_level);
    }
  }, [exposureParams]);

  if (!exposureParams) {
    return (
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        loading exposure params…
      </p>
    );
  }

  const isManual = exposureParams.mode === 0;

  const debouncedSet = (field: string, v: number) => {
    dragging.current = true;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      dragging.current = false;
      setModuleParam("exposure", { [field]: v });
    }, 150);
  };

  const biasEv = exposureParams.exposure_bias_ev;
  const hlBias = exposureParams.highlight_bias_ev;

  return (
    <>
      <BauhausCombo
        label="mode"
        options={["manual", "automatic"]}
        value={isManual ? "manual" : "automatic"}
        onChange={(v) =>
          setModuleParam("exposure", { mode: v === "manual" ? 0 : 1 })
        }
      />

      {isManual ? (
        <>
          <BauhausCheckbox
            label={`compensate camera exposure (${biasEv >= 0 ? "+" : ""}${biasEv.toFixed(1)} EV)`}
            checked={exposureParams.compensate_exposure_bias}
            align="left"
            onChange={(checked) =>
              setModuleParam("exposure", { compensate_exposure_bias: checked })
            }
          />

          {hlBias > 0 && (
            <BauhausCheckbox
              label={`highlight preservation mode (${hlBias.toFixed(1)} EV)`}
              checked={exposureParams.compensate_hilite_pres}
              align="left"
              onChange={(checked) =>
                setModuleParam("exposure", { compensate_hilite_pres: checked })
              }
            />
          )}

          <BauhausSlider
            label="exposure"
            value={localExposure}
            min={-3}
            max={4}
            step={0.001}
            defaultValue={0}
            origin={0}
            format={(v) => `${v >= 0 ? "+" : ""}${v.toFixed(3)} EV`}
            onChange={(v) => { setLocalExposure(v); debouncedSet("exposure", v); }}
          />
        </>
      ) : (
        <>
          <BauhausSlider
            label="percentile"
            value={localPercentile}
            min={0}
            max={100}
            step={0.01}
            defaultValue={50}
            format={(v) => `${v.toFixed(2)}%`}
            onChange={(v) => { setLocalPercentile(v); debouncedSet("deflicker_percentile", v); }}
          />

          <BauhausSlider
            label="target level"
            value={localTarget}
            min={-18}
            max={18}
            step={0.01}
            defaultValue={-4}
            format={(v) => `${v.toFixed(2)} EV`}
            onChange={(v) => { setLocalTarget(v); debouncedSet("deflicker_target_level", v); }}
          />
        </>
      )}

      <BauhausSlider
        label="black level correction"
        value={localBlack}
        min={-0.1}
        max={0.1}
        step={0.0001}
        defaultValue={0}
        origin={0}
        format={(v) => v.toFixed(4)}
        onChange={(v) => { setLocalBlack(v); debouncedSet("black", v); }}
      />
    </>
  );
}
