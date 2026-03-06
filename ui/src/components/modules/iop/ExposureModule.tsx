import { useState, useEffect, useRef, useCallback } from "react";
import { useDevelopStore } from "../../../stores/developStore";
import BauhausSlider from "../../controls/BauhausSlider";
import BauhausCheckbox from "../../controls/BauhausCheckbox";
import BauhausCombo from "../../controls/BauhausCombo";
import BauhausLabel from "../../controls/BauhausLabel";
import BauhausPicker from "../../controls/BauhausPicker";
import { usePickerStore } from "../../../stores/pickerStore";

/**
 * Throttled param applicator: sends set_params as fast as the IPC allows.
 * The server processes the pipeline asynchronously and pushes preview_ready
 * events when SHM is written — the store event listener handles frame fetch.
 */
function useThrottledParam(op: string) {
  const applyParam = useDevelopStore((s) => s.applyParam);
  const commitParam = useDevelopStore((s) => s.commitParam);
  const fetchHistory = useDevelopStore((s) => s.fetchHistory);
  const fetchGenericParams = useDevelopStore((s) => s.fetchGenericParams);
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
        await commitParam(op);
        fetchHistory();
        fetchGenericParams(op);
      }
    },
    [op, applyParam, commitParam, fetchHistory, fetchGenericParams],
  );

  return { apply, draggingRef };
}

/** Convert sRGB 0-1 to linear */
function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

export default function ExposureModule() {
  const params = useDevelopStore((s) => s.genericParams["exposure"]);
  const setModuleParam = useDevelopStore((s) => s.setModuleParam);
  const fetchGenericParams = useDevelopStore((s) => s.fetchGenericParams);
  const samplePixels = useDevelopStore((s) => s.samplePixels);

  const [localExposure, setLocalExposure] = useState(0);
  const [localBlack, setLocalBlack] = useState(0);
  const [localPercentile, setLocalPercentile] = useState(50);
  const [localTarget, setLocalTarget] = useState(-4);

  const { apply: throttledApply, draggingRef } = useThrottledParam("exposure");
  const pickerActive = usePickerStore((s) => s.active);
  const pickerBox = usePickerStore((s) => s.box);
  const deactivatePicker = usePickerStore((s) => s.deactivate);

  // Deactivate picker when module unmounts (collapsed / session closed)
  useEffect(() => {
    return () => deactivatePicker("exposure-picker");
  }, [deactivatePicker]);

  // Track current exposure in a ref to avoid effect loops
  const exposureRef = useRef((params?.exposure as number) ?? 0);
  useEffect(() => {
    if (params) exposureRef.current = params.exposure as number;
  }, [params]);

  // When picker box changes, sample pixels and compute exposure correction
  const pickerBoxKey = `${pickerBox.x},${pickerBox.y},${pickerBox.w},${pickerBox.h}`;
  useEffect(() => {
    if (pickerActive?.id !== "exposure-picker") return;

    let cancelled = false;
    (async () => {
      const sample = await samplePixels(pickerBox.x, pickerBox.y, pickerBox.w, pickerBox.h);
      if (cancelled || !sample) return;

      // Convert sampled sRGB to linear luminance
      const rLin = srgbToLinear(sample.mean_r);
      const gLin = srgbToLinear(sample.mean_g);
      const bLin = srgbToLinear(sample.mean_b);
      const Y = 0.2126 * rLin + 0.7152 * gLin + 0.0722 * bLin;

      if (Y <= 1e-6) return; // avoid log2(0)

      // Target: middle gray (18% reflectance = ~0.184 linear)
      const targetY = 0.184;
      const deltaEV = Math.log2(targetY / Y);
      const newExposure = Math.max(-3, Math.min(4, exposureRef.current + deltaEV));

      setLocalExposure(newExposure);
      setModuleParam("exposure", { exposure: newExposure });
    })();

    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pickerActive, pickerBoxKey, samplePixels, setModuleParam]);

  // Fetch params on mount if not already loaded
  useEffect(() => {
    if (!params) {
      fetchGenericParams("exposure");
    }
  }, [params, fetchGenericParams]);

  // Sync local slider state from store when not dragging
  useEffect(() => {
    if (params && !draggingRef.current) {
      setLocalExposure(params.exposure as number);
      setLocalBlack(params.black as number);
      setLocalPercentile(params.deflicker_percentile as number);
      setLocalTarget(params.deflicker_target_level as number);
    }
  }, [params, draggingRef]);

  if (!params) {
    return (
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        loading exposure params…
      </p>
    );
  }

  const isManual = params.mode === 0;
  const biasEv = (params.exposure_bias_ev as number) ?? 0;
  const hlBias = (params.highlight_bias_ev as number) ?? 0;

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
            checked={params.compensate_exposure_bias as boolean}
            align="left"
            onChange={(checked) =>
              setModuleParam("exposure", { compensate_exposure_bias: checked })
            }
          />

          {hlBias > 0 && (
            <BauhausCheckbox
              label={`highlight preservation mode (${hlBias.toFixed(1)} EV)`}
              checked={params.compensate_hilite_pres as boolean}
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
            actionIcon={<BauhausPicker id="exposure-picker" module="exposure" mode="area" />}
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

          <BauhausLabel
            label="computed EC"
            value={
              (params.deflicker_computed_exposure as number) != null
                ? `${(params.deflicker_computed_exposure as number) >= 0 ? "+" : ""}${(params.deflicker_computed_exposure as number).toFixed(2)} EV`
                : "N/A"
            }
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
