import { useState, useEffect, useRef, useCallback } from "react";
import { useDevelopStore } from "../../../stores/developStore";
import BauhausSlider from "../../controls/BauhausSlider";
import BauhausCombo from "../../controls/BauhausCombo";
import BauhausCheckbox from "../../controls/BauhausCheckbox";
import BauhausSection from "../../controls/BauhausSection";

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

// Demosaic method enum values (must match server-side)
const DEMOSAIC_DUAL = 2048;
const DEMOSAIC_XTRANS = 1024;

// Bayer methods
const BAYER_METHODS = [
  { value: 0, label: "PPG" },
  { value: 1, label: "AMaZE" },
  { value: 2, label: "VNG4" },
  { value: 5, label: "RCD" },
  { value: 6, label: "LMMSE" },
  { value: DEMOSAIC_DUAL | 5, label: "RCD (dual)" },
  { value: DEMOSAIC_DUAL | 1, label: "AMaZE (dual)" },
  { value: 3, label: "passthrough (monochrome)" },
  { value: 4, label: "photosite color (debug)" },
];

// X-Trans methods
const XTRANS_METHODS = [
  { value: DEMOSAIC_XTRANS | 0, label: "VNG" },
  { value: DEMOSAIC_XTRANS | 1, label: "Markesteijn 1-pass" },
  { value: DEMOSAIC_XTRANS | 2, label: "Markesteijn 3-pass" },
  { value: DEMOSAIC_XTRANS | 4, label: "frequency domain chroma" },
  { value: DEMOSAIC_DUAL | DEMOSAIC_XTRANS | 2, label: "Markesteijn 3-pass (dual)" },
  { value: DEMOSAIC_XTRANS | 3, label: "passthrough (monochrome)" },
  { value: DEMOSAIC_XTRANS | 5, label: "photosite color (debug)" },
];

// Bayer4 methods (limited)
const BAYER4_METHODS = [
  { value: 2, label: "VNG4" },
  { value: 3, label: "passthrough (monochrome)" },
  { value: 4, label: "photosite color (debug)" },
];

// Mono methods
const MONO_METHODS = [
  { value: 3, label: "passthrough (monochrome)" },
  { value: 7, label: "Monochrome" },
];

const GREEN_EQ_OPTIONS = ["disabled", "local average", "full average", "full and local average"];
const COLOR_SMOOTH_OPTIONS = ["disabled", "once", "twice", "three times", "four times", "five times"];
const LMMSE_REFINE_OPTIONS = ["basic", "median", "3x median", "refine & medians", "2x refine + medians"];

function getMethodsForSensor(sensor: string) {
  switch (sensor) {
    case "xtrans": return XTRANS_METHODS;
    case "bayer4": return BAYER4_METHODS;
    case "mono": return MONO_METHODS;
    default: return BAYER_METHODS;
  }
}

export default function DemosaicModule() {
  const params = useDevelopStore((s) => s.genericParams["demosaic"]);
  const setModuleParam = useDevelopStore((s) => s.setModuleParam);
  const fetchGenericParams = useDevelopStore((s) => s.fetchGenericParams);

  const [localMedianThrs, setLocalMedianThrs] = useState(0);
  const [localDualThrs, setLocalDualThrs] = useState(0.2);
  const [localCsRadius, setLocalCsRadius] = useState(0);
  const [localCsThrs, setLocalCsThrs] = useState(0.4);
  const [localCsBoost, setLocalCsBoost] = useState(0);
  const [localCsIter, setLocalCsIter] = useState(8);
  const [localCsCenter, setLocalCsCenter] = useState(0);

  const { apply: throttledApply, draggingRef } = useThrottledParam("demosaic");

  useEffect(() => {
    if (!params) {
      fetchGenericParams("demosaic");
    }
  }, [params, fetchGenericParams]);

  useEffect(() => {
    if (params && !draggingRef.current) {
      setLocalMedianThrs(params.median_thrs as number);
      setLocalDualThrs(params.dual_thrs as number);
      setLocalCsRadius(params.cs_radius as number);
      setLocalCsThrs(params.cs_thrs as number);
      setLocalCsBoost(params.cs_boost as number);
      setLocalCsIter(params.cs_iter as number);
      setLocalCsCenter(params.cs_center as number);
    }
  }, [params, draggingRef]);

  if (!params) {
    return (
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        loading demosaic params…
      </p>
    );
  }

  const sensor = params.sensor_type as string;
  const methods = getMethodsForSensor(sensor);
  const method = params.demosaicing_method as number;
  const isDual = !!(method & DEMOSAIC_DUAL);
  const isLmmse = method === 6;
  const isPpg = method === 0;
  const isBayer = sensor === "bayer";
  const isMono = sensor === "mono";
  const isBayer4 = sensor === "bayer4";
  const isXtrans = sensor === "xtrans";
  const isPassing = method === 3 || method === 4
    || method === (DEMOSAIC_XTRANS | 3) || method === (DEMOSAIC_XTRANS | 5);

  const captureSupport = !isPassing && !isBayer4;
  const showCapture = captureSupport && params.cs_enabled as boolean;

  if (sensor === "mono" && !methods.some(m => m.value === method)) {
    return (
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        not applicable
      </p>
    );
  }

  const currentMethodLabel = methods.find(m => m.value === method)?.label ?? "unknown";

  return (
    <>
      <BauhausCombo
        label="method"
        options={methods.map(m => m.label)}
        value={currentMethodLabel}
        onChange={(label) => {
          const m = methods.find(x => x.label === label);
          if (m) setModuleParam("demosaic", { demosaicing_method: m.value });
        }}
      />

      {isBayer && isPpg && (
        <BauhausSlider
          label="edge threshold"
          value={localMedianThrs}
          min={0}
          max={1}
          step={0.001}
          defaultValue={0}
          format={(v) => v.toFixed(3)}
          onChange={(v) => { setLocalMedianThrs(v); throttledApply("median_thrs", v); }}
        />
      )}

      {isDual && !isBayer4 && !isMono && (
        <BauhausSlider
          label="dual threshold"
          value={localDualThrs}
          min={0}
          max={1}
          step={0.01}
          defaultValue={0.2}
          format={(v) => v.toFixed(2)}
          onChange={(v) => { setLocalDualThrs(v); throttledApply("dual_thrs", v); }}
        />
      )}

      {isLmmse && (
        <BauhausCombo
          label="LMMSE refine"
          options={LMMSE_REFINE_OPTIONS}
          value={LMMSE_REFINE_OPTIONS[params.lmmse_refine as number] ?? "basic"}
          onChange={(label) => {
            const idx = LMMSE_REFINE_OPTIONS.indexOf(label);
            if (idx >= 0) setModuleParam("demosaic", { lmmse_refine: idx });
          }}
        />
      )}

      {!isPassing && !isBayer4 && !isDual && !isMono && (
        <BauhausCombo
          label="color smoothing"
          options={COLOR_SMOOTH_OPTIONS}
          value={COLOR_SMOOTH_OPTIONS[params.color_smoothing as number] ?? "disabled"}
          onChange={(label) => {
            const idx = COLOR_SMOOTH_OPTIONS.indexOf(label);
            if (idx >= 0) setModuleParam("demosaic", { color_smoothing: idx });
          }}
        />
      )}

      {!isPassing && !isBayer4 && !isXtrans && !isMono && (
        <BauhausCombo
          label="match greens"
          options={GREEN_EQ_OPTIONS}
          value={GREEN_EQ_OPTIONS[params.green_eq as number] ?? "disabled"}
          onChange={(label) => {
            const idx = GREEN_EQ_OPTIONS.indexOf(label);
            if (idx >= 0) setModuleParam("demosaic", { green_eq: idx });
          }}
        />
      )}

      {captureSupport && (
        <BauhausCheckbox
          label="capture sharpen"
          checked={params.cs_enabled as boolean}
          align="left"
          onChange={(checked) => setModuleParam("demosaic", { cs_enabled: checked })}
        />
      )}

      {showCapture && (
        <BauhausSection title="capture sharpen controls">
          <BauhausSlider
            label="iterations"
            value={localCsIter}
            min={1}
            max={25}
            step={1}
            defaultValue={8}
            format={(v) => `${Math.round(v)}`}
            onChange={(v) => { setLocalCsIter(v); throttledApply("cs_iter", v); }}
          />
          <BauhausSlider
            label="radius"
            value={localCsRadius}
            min={0}
            max={1.5}
            step={0.01}
            defaultValue={0}
            format={(v) => `${v.toFixed(2)} px`}
            onChange={(v) => { setLocalCsRadius(v); throttledApply("cs_radius", v); }}
          />
          <BauhausSlider
            label="contrast sensitivity"
            value={localCsThrs}
            min={0}
            max={1}
            step={0.001}
            defaultValue={0.4}
            format={(v) => v.toFixed(3)}
            onChange={(v) => { setLocalCsThrs(v); throttledApply("cs_thrs", v); }}
          />
          <BauhausSlider
            label="corner boost"
            value={localCsBoost}
            min={0}
            max={1.5}
            step={0.01}
            defaultValue={0}
            format={(v) => `${v.toFixed(2)} px`}
            onChange={(v) => { setLocalCsBoost(v); throttledApply("cs_boost", v); }}
          />
          {(params.cs_boost as number) > 0 && (
            <BauhausSlider
              label="sharp center"
              value={localCsCenter}
              min={0}
              max={1}
              step={0.01}
              defaultValue={0}
              format={(v) => `${Math.round(v * 100)}%`}
              onChange={(v) => { setLocalCsCenter(v); throttledApply("cs_center", v); }}
            />
          )}
        </BauhausSection>
      )}
    </>
  );
}
