import { useState, useEffect, useCallback, useMemo } from "react";
import { Aperture, Pipette, Pen, Camera, SwitchCamera, AlertTriangle } from "lucide-react";
import { useDevelopStore, getEnabledOps } from "../../../stores/developStore";
import { useThrottledParam } from "../../../hooks/useThrottledParam";
import { usePickerStore } from "../../../stores/pickerStore";
import { useIopModuleContext } from "../IopModuleContext";
import BauhausSlider from "../../controls/BauhausSlider";
import BauhausButton from "../../controls/BauhausButton";
import BauhausTooltip from "../../controls/BauhausTooltip";
import BauhausCombo from "../../controls/BauhausCombo";
import BauhausSection from "../../controls/BauhausSection";

const PRESET_OPTIONS = [
  "as shot",
  "from image area",
  "user modified",
  "camera reference",
  "as shot to reference",
];

/** Convert sRGB 0-1 to linear */
function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** Linear to sRGB gamma */
function linearToSrgb(c: number): number {
  if (c <= 0) return 0;
  if (c >= 1) return 1;
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1.0 / 2.4) - 0.055;
}

/**
 * Attempt CIE daylight illuminant chromaticity from temperature.
 * Simplified from darktable's spectral approach — good enough for gradient display.
 * Uses CIE daylight locus (for >= 4000K) and Planckian locus (below).
 */
function tempToXY(tempK: number): [number, number] {
  const T = tempK;
  let x: number;
  if (T >= 4000 && T <= 7000) {
    x = -4.6070e9 / (T * T * T) + 2.9678e6 / (T * T) + 0.09911e3 / T + 0.244063;
  } else if (T > 7000 && T <= 25000) {
    x = -2.0064e9 / (T * T * T) + 1.9018e6 / (T * T) + 0.24748e3 / T + 0.237040;
  } else {
    // Planckian locus approximation for low temps
    x = -0.2661239e9 / (T * T * T) - 0.2343589e6 / (T * T) + 0.8776956e3 / T + 0.179910;
  }
  let y: number;
  if (T >= 4000) {
    if (x <= 0.2) {
      y = -0.20219683 + 2.18555832 * x - 1.34811020 * x * x;
    } else {
      y = -3.000 * x * x + 2.870 * x - 0.275;
    }
  } else {
    y = -1.1063814 * x * x * x - 1.34811020 * x * x + 2.18555832 * x - 0.20219683;
  }
  return [x, Math.max(y, 0.01)];
}

/** CIE xy + tint → approximate linear RGB for gradient visualization */
function tempTintToRGB(tempK: number, tint: number): [number, number, number] {
  const [x, y] = tempToXY(tempK);
  // Apply tint as green-magenta shift: scale Y relative to X+Z
  const X = x / y;
  const Y = 1.0;
  const Z = (1.0 - x - y) / y;

  // Tint shifts green channel: tint > 1 = more green, < 1 = more magenta
  // Model as scaling the Y (luminance/green) component
  const tintedX = X;
  const tintedY = Y * tint;
  const tintedZ = Z;

  // XYZ to linear sRGB (D65)
  let r =  3.2404542 * tintedX - 1.5371385 * tintedY - 0.4985314 * tintedZ;
  let g = -0.9692660 * tintedX + 1.8760108 * tintedY + 0.0415560 * tintedZ;
  let b =  0.0556434 * tintedX - 0.2040259 * tintedY + 1.0572252 * tintedZ;

  // Normalize to max = 1
  const mx = Math.max(r, g, b, 0.001);
  r /= mx; g /= mx; b /= mx;

  return [linearToSrgb(r), linearToSrgb(g), linearToSrgb(b)];
}

function rgbToCSS(r: number, g: number, b: number): string {
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}

/** Build a CSS gradient string sampling a parameter range */
function buildGradient(
  stops: number,
  min: number,
  max: number,
  colorFn: (v: number) => [number, number, number],
): string {
  const colors: string[] = [];
  for (let i = 0; i < stops; i++) {
    const t = i / (stops - 1);
    const v = min + t * (max - min);
    const [r, g, b] = colorFn(v);
    colors.push(rgbToCSS(r, g, b));
  }
  return `linear-gradient(to right, ${colors.join(", ")})`;
}

export default function TemperatureModule() {
  const { setIndicator } = useIopModuleContext();
  const params = useDevelopStore((s) => s.genericParams["temperature"]);
  const setModuleParam = useDevelopStore((s) => s.setModuleParam);
  const fetchGenericParams = useDevelopStore((s) => s.fetchGenericParams);
  const samplePixels = useDevelopStore((s) => s.samplePixels);
  const historyItems = useDevelopStore((s) => s.historyItems);
  const isMandatory = historyItems.some((h) => h.op === "temperature" && h.mandatory);

  const [localRed, setLocalRed] = useState(1);
  const [localGreen, setLocalGreen] = useState(1);
  const [localBlue, setLocalBlue] = useState(1);
  const [localTempK, setLocalTempK] = useState(5000);
  const [localTint, setLocalTint] = useState(1);

  const { apply: throttledApply, release: throttledRelease, draggingRef } = useThrottledParam("temperature");

  // Callback ref: when the trouble banner mounts, scroll parent to compensate
  const troubleRefCb = useCallback((el: HTMLParagraphElement | null) => {
    if (!el) return;
    const scroller = el.closest(".overflow-y-auto");
    if (scroller) {
      scroller.scrollTop += el.offsetHeight + 4;
    }
  }, []);

  const pickerActive = usePickerStore((s) => s.active);
  const pickerBox = usePickerStore((s) => s.box);
  const activatePicker = usePickerStore((s) => s.activate);
  const deactivatePicker = usePickerStore((s) => s.deactivate);

  // Deactivate picker when module unmounts
  useEffect(() => {
    return () => deactivatePicker("temperature-picker");
  }, [deactivatePicker]);

  // When picker box changes, sample pixels and compute WB coefficients
  const pickerBoxKey = `${pickerBox.x},${pickerBox.y},${pickerBox.w},${pickerBox.h}`;
  useEffect(() => {
    if (pickerActive?.id !== "temperature-picker") return;

    let cancelled = false;
    (async () => {
      const sample = await samplePixels(pickerBox.x, pickerBox.y, pickerBox.w, pickerBox.h);
      if (cancelled || !sample) return;

      // Convert sampled sRGB (output, post-WB) to linear
      const rLin = srgbToLinear(sample.mean_r);
      const gLin = srgbToLinear(sample.mean_g);
      const bLin = srgbToLinear(sample.mean_b);

      if (rLin < 1e-6 || gLin < 1e-6 || bLin < 1e-6) return;

      // The sampled area should be neutral. Adjust current coefficients
      // by the inverse of the color cast: if output is too red, reduce red coeff.
      const curRed = (params?.red as number) ?? 1;
      const curGreen = (params?.green as number) ?? 1;
      const curBlue = (params?.blue as number) ?? 1;

      const corrRed = curRed * (gLin / rLin);
      const corrBlue = curBlue * (gLin / bLin);
      // Normalize so green = current green
      const newRed = corrRed;
      const newGreen = curGreen;
      const newBlue = corrBlue;

      setLocalRed(newRed);
      setLocalGreen(newGreen);
      setLocalBlue(newBlue);
      setModuleParam("temperature", { red: newRed, green: newGreen, blue: newBlue });
    })();

    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pickerActive, pickerBoxKey, samplePixels, setModuleParam]);

  useEffect(() => {
    if (!params) {
      fetchGenericParams("temperature");
    }
  }, [params, fetchGenericParams]);

  useEffect(() => {
    if (params && !draggingRef.current) {
      setLocalRed(params.red as number);
      setLocalGreen(params.green as number);
      setLocalBlue(params.blue as number);
      if (params.temperature_k != null) setLocalTempK(params.temperature_k as number);
      if (params.tint != null) setLocalTint(params.tint as number);
    }
  }, [params, draggingRef]);

  const tempGradient = useMemo(
    () => buildGradient(9, 1901, 25000, (t) => tempTintToRGB(t, localTint)),
    [localTint],
  );
  const tintGradient = useMemo(
    () => buildGradient(9, 0.135, 2.326, (ti) => tempTintToRGB(localTempK, ti)),
    [localTempK],
  );

  if (isMandatory) {
    return (
      <p className="module-warning">white balance disabled for camera</p>
    );
  }

  if (!params) {
    return (
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        loading white balance params...
      </p>
    );
  }

  const presetIdx = params.preset as number;
  const presetLabel = PRESET_OPTIONS[presetIdx] ?? PRESET_OPTIONS[0];

  // Detect "white balance applied twice": color calibration is doing chromatic
  // adaptation while temperature is not set to camera reference (D65).
  // preset 3 = D65, preset 4 = D65_LATE (as shot to reference)
  const enabledOps = getEnabledOps(historyItems);
  const colorCalEnabled = enabledOps.has("channelmixerrgb");
  const tempEnabled = enabledOps.has("temperature");
  const isD65 = presetIdx === 3 || presetIdx === 4;
  const wbAppliedTwice = tempEnabled && colorCalEnabled && !isD65;

  useEffect(() => {
    setIndicator(wbAppliedTwice
      ? <BauhausTooltip content="white balance applied twice" placement="left">
          <span className="module-trouble-icon"><AlertTriangle size={12} /></span>
        </BauhausTooltip>
      : null
    );
  }, [wbAppliedTwice, setIndicator]);

  const presetButtons = [
    { preset: 0, icon: <Aperture size={12} />, title: "as shot" },
    { preset: 1, icon: <Pipette size={12} />, title: "from image area" },
    { preset: 2, icon: <Pen size={12} />, title: "user modified" },
    { preset: 3, icon: <Camera size={12} />, title: "camera reference" },
    { preset: 4, icon: <SwitchCamera size={12} />, title: "as shot to reference" },
  ];

  return (
    <>
      {wbAppliedTwice && (
        <p ref={troubleRefCb} className="module-trouble" title={
          "the color calibration module is enabled and already provides\n"
          + "chromatic adaptation.\n"
          + "set the white balance here to camera reference (D65)\n"
          + "or disable chromatic adaptation in color calibration."
        }>
          white balance applied twice
        </p>
      )}
      <div className="bauhaus-button-row">
        {presetButtons.map(({ preset, icon, title }) => (
          <BauhausButton
            key={preset}
            icon={icon}
            title={title}
            active={preset === 1 ? pickerActive?.id === "temperature-picker" : presetIdx === preset}
            transparent
            onClick={() => {
              if (preset === 1) {
                activatePicker("temperature-picker", "temperature", "area");
              } else {
                setModuleParam("temperature", { preset });
              }
            }}
          />
        ))}
      </div>

      <BauhausCombo
        label="settings"
        options={PRESET_OPTIONS}
        value={presetLabel}
        onChange={(label) => {
          const idx = PRESET_OPTIONS.indexOf(label);
          if (idx === 1) {
            activatePicker("temperature-picker", "temperature", "area");
          } else if (idx >= 0) {
            setModuleParam("temperature", { preset: idx });
          }
        }}
      />

      <BauhausSlider
        label="temperature"
        value={localTempK}
        min={1901}
        max={25000}
        step={1}
        defaultValue={5000}
        gradient={tempGradient}
        format={(v) => `${Math.round(v)} K`}
        onChange={(v) => { setLocalTempK(v); throttledApply("temperature_k", v); }}
        onRelease={throttledRelease}
      />

      <BauhausSlider
        label="tint"
        value={localTint}
        min={0.135}
        max={2.326}
        step={0.001}
        defaultValue={1}
        gradient={tintGradient}
        format={(v) => v.toFixed(3)}
        onChange={(v) => { setLocalTint(v); throttledApply("tint", v); }}
        onRelease={throttledRelease}
      />

      <BauhausSection title="channel coefficients">
        <BauhausSlider
          label="red"
          value={localRed}
          min={0}
          max={8}
          step={0.001}
          defaultValue={1}
          color="#ff4040"
          format={(v) => v.toFixed(4)}
          onChange={(v) => { setLocalRed(v); throttledApply("red", v); }}
          onRelease={throttledRelease}
        />

        <BauhausSlider
          label="green"
          value={localGreen}
          min={0}
          max={8}
          step={0.001}
          defaultValue={1}
          color="#40c040"
          format={(v) => v.toFixed(4)}
          onChange={(v) => { setLocalGreen(v); throttledApply("green", v); }}
          onRelease={throttledRelease}
        />

        <BauhausSlider
          label="blue"
          value={localBlue}
          min={0}
          max={8}
          step={0.001}
          defaultValue={1}
          color="#4060ff"
          format={(v) => v.toFixed(4)}
          onChange={(v) => { setLocalBlue(v); throttledApply("blue", v); }}
          onRelease={throttledRelease}
        />
      </BauhausSection>
    </>
  );
}
