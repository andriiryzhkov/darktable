import { useState, useEffect, useRef, useCallback } from "react";
import { useDevelopStore } from "../../../stores/developStore";
import BauhausSlider from "../../controls/BauhausSlider";
import BauhausCheckbox from "../../controls/BauhausCheckbox";
import BauhausCombo from "../../controls/BauhausCombo";

/**
 * Throttled param applicator: sends set_params as fast as the IPC allows.
 * The server processes the pipeline asynchronously and pushes preview_ready
 * events when SHM is written — the store event listener handles frame fetch.
 */
function useThrottledParam(op: string) {
  const applyParam = useDevelopStore((s) => s.applyParam);
  const commitParam = useDevelopStore((s) => s.commitParam);
  const fetchHistory = useDevelopStore((s) => s.fetchHistory);
  const fetchModuleParams = useDevelopStore((s) => s.fetchModuleParams);
  const busyRef = useRef(false);
  const pendingRef = useRef<Record<string, unknown> | null>(null);
  const draggingRef = useRef(false);

  const apply = useCallback(
    async (field: string, v: number) => {
      draggingRef.current = true;
      pendingRef.current = { [field]: v };
      if (busyRef.current) return;

      busyRef.current = true;
      try {
        while (pendingRef.current) {
          const params = pendingRef.current;
          pendingRef.current = null;
          await applyParam(op, params);
        }
      } finally {
        busyRef.current = false;
        draggingRef.current = false;
        // Commit final value to history, then sync UI
        await commitParam(op);
        fetchHistory();
        fetchModuleParams(op);
      }
    },
    [op, applyParam, commitParam, fetchHistory, fetchModuleParams],
  );

  return { apply, draggingRef };
}

export default function ExposureModule() {
  const exposureParams = useDevelopStore((s) => s.exposureParams);
  const setModuleParam = useDevelopStore((s) => s.setModuleParam);
  const fetchModuleParams = useDevelopStore((s) => s.fetchModuleParams);

  const [localExposure, setLocalExposure] = useState(0);
  const [localBlack, setLocalBlack] = useState(0);
  const [localPercentile, setLocalPercentile] = useState(50);
  const [localTarget, setLocalTarget] = useState(-4);

  const { apply: throttledApply, draggingRef } = useThrottledParam("exposure");

  // Fetch params on mount if not already loaded
  useEffect(() => {
    if (!exposureParams) {
      fetchModuleParams("exposure");
    }
  }, [exposureParams, fetchModuleParams]);

  // Sync local slider state from store when not dragging
  useEffect(() => {
    if (exposureParams && !draggingRef.current) {
      setLocalExposure(exposureParams.exposure);
      setLocalBlack(exposureParams.black);
      setLocalPercentile(exposureParams.deflicker_percentile);
      setLocalTarget(exposureParams.deflicker_target_level);
    }
  }, [exposureParams, draggingRef]);

  if (!exposureParams) {
    return (
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        loading exposure params…
      </p>
    );
  }

  const isManual = exposureParams.mode === 0;
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
            onChange={(v) => { setLocalExposure(v); throttledApply("exposure", v); }}
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
            onChange={(v) => { setLocalPercentile(v); throttledApply("deflicker_percentile", v); }}
          />

          <BauhausSlider
            label="target level"
            value={localTarget}
            min={-18}
            max={18}
            step={0.01}
            defaultValue={-4}
            format={(v) => `${v.toFixed(2)} EV`}
            onChange={(v) => { setLocalTarget(v); throttledApply("deflicker_target_level", v); }}
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
        onChange={(v) => { setLocalBlack(v); throttledApply("black", v); }}
      />
    </>
  );
}
