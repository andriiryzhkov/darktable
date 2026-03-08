import { useState, useEffect, useRef, useCallback } from "react";
import { useDevelopStore } from "../../../stores/developStore";
import { useThrottledParam } from "../../../hooks/useThrottledParam";
import BauhausSlider from "../../controls/BauhausSlider";
import BauhausCombo from "../../controls/BauhausCombo";
import BauhausCheckbox from "../../controls/BauhausCheckbox";
import BauhausTabBar from "../../controls/BauhausTabBar";

// --- Enums matching darktable C code ---

const ILLUMINANT_LABELS = [
  "same as pipeline (D50)",  // 0 PIPE
  "A (incandescent)",        // 1 A
  "D (daylight)",            // 2 D
  "E (equi-energy)",         // 3 E
  "F (fluorescent)",         // 4 F
  "LED",                     // 5 LED
  "Planckian (black body)",  // 6 BB
  "custom",                  // 7 CUSTOM
  "detect (surfaces)",       // 8 DETECT_SURFACES
  "detect (edges)",          // 9 DETECT_EDGES
  "as shot in camera",       // 10 CAMERA
];
const ILLUMINANT_IDX_D = 2;
const ILLUMINANT_IDX_F = 4;
const ILLUMINANT_IDX_LED = 5;
const ILLUMINANT_IDX_BB = 6;
const ILLUMINANT_IDX_CUSTOM = 7;

const FLUO_LABELS = [
  "F1 (daylight 6430 K)",
  "F2 (cool white 4230 K)",
  "F3 (white 3450 K)",
  "F4 (warm white 2940 K)",
  "F5 (daylight 6350 K)",
  "F6 (lite white 4150 K)",
  "F7 (D65 simulator 6500 K)",
  "F8 (D50 simulator 5000 K)",
  "F9 (cool white deluxe 4150 K)",
  "F10 (tuned RGB 5000 K)",
  "F11 (tuned RGB 4000 K)",
  "F12 (tuned RGB 3000 K)",
];

const LED_LABELS = [
  "B1 (2733 K)",
  "B2 (2998 K)",
  "B3 (4103 K)",
  "B4 (5109 K)",
  "B5 (6598 K)",
  "BH1 (2851 K)",
  "RGB1 (2840 K)",
  "V1 (2724 K)",
  "V2 (4070 K)",
];

const ADAPTATION_LABELS = [
  "linear Bradford",
  "CAT16 (CIECAM16)",
  "non-linear Bradford",
  "XYZ",
  "none (bypass)",
];

const VERSION_LABELS = [
  "v1 (2020)",
  "v2 (2021)",
  "v3 (2021)",
];

const TAB_NAMES = ["CAT", "R", "G", "B", "colorfulness", "brightness", "gray"] as const;
type TabName = (typeof TAB_NAMES)[number];

// --- Throttled param hook ---

// Helper: update one element of a float[4] array param
function arrSet(arr: number[], idx: number, v: number): number[] {
  const copy = [...arr];
  copy[idx] = v;
  return copy;
}

// --- Component ---

export default function ChannelMixerRgbModule() {
  const params = useDevelopStore((s) => s.genericParams["channelmixerrgb"]);
  const setModuleParam = useDevelopStore((s) => s.setModuleParam);
  const fetchGenericParams = useDevelopStore((s) => s.fetchGenericParams);

  const [tab, setTab] = useState<TabName>("CAT");

  // Local slider state for responsive dragging
  const [localTemp, setLocalTemp] = useState(5003);
  const [localGamut, setLocalGamut] = useState(1.0);
  const [localRed, setLocalRed] = useState([1, 0, 0, 0]);
  const [localGreen, setLocalGreen] = useState([0, 1, 0, 0]);
  const [localBlue, setLocalBlue] = useState([0, 0, 1, 0]);
  const [localSat, setLocalSat] = useState([0, 0, 0, 0]);
  const [localLight, setLocalLight] = useState([0, 0, 0, 0]);
  const [localGrey, setLocalGrey] = useState([0, 0, 0, 0]);

  const { apply: throttledApply, release: throttledRelease, draggingRef } = useThrottledParam("channelmixerrgb");

  useEffect(() => {
    if (!params) fetchGenericParams("channelmixerrgb");
  }, [params, fetchGenericParams]);

  useEffect(() => {
    if (params && !draggingRef.current) {
      setLocalTemp(params.temperature as number);
      setLocalGamut(params.gamut as number);
      setLocalRed(params.red as number[]);
      setLocalGreen(params.green as number[]);
      setLocalBlue(params.blue as number[]);
      setLocalSat(params.saturation as number[]);
      setLocalLight(params.lightness as number[]);
      setLocalGrey(params.grey as number[]);
    }
  }, [params, draggingRef]);

  if (!params) {
    return (
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        loading color calibration params...
      </p>
    );
  }

  const illuminant = params.illuminant as number;
  const adaptation = params.adaptation as number;
  const showTemp = illuminant === ILLUMINANT_IDX_D || illuminant === ILLUMINANT_IDX_BB;
  const showFluo = illuminant === ILLUMINANT_IDX_F;
  const showLed = illuminant === ILLUMINANT_IDX_LED;
  const showXY = illuminant === ILLUMINANT_IDX_CUSTOM;

  // Channel mixing tabs share this pattern
  const renderMixerTab = (
    arr: number[],
    setArr: (v: number[]) => void,
    field: string,
    normalizeField: string,
    normalizeValue: boolean,
    color?: [string, string, string],
  ) => (
    <>
      {[0, 1, 2].map((i) => {
        const chLabel = ["input R", "input G", "input B"][i];
        return (
          <BauhausSlider
            key={i}
            label={chLabel}
            value={arr[i]}
            min={-2}
            max={2}
            step={0.001}
            defaultValue={field === "grey" ? (i === 1 ? 1 : 0) : (i === ["red", "green", "blue"].indexOf(field) ? 1 : 0)}
            origin={0}
            color={color?.[i]}
            format={(v) => `${v >= 0 ? "+" : ""}${v.toFixed(3)}`}
            onChange={(v) => {
              const next = arrSet(arr, i, v);
              setArr(next);
              throttledApply(field, next);
            }}
            onRelease={throttledRelease}
          />
        );
      })}
      <BauhausCheckbox
        label="normalize channel"
        checked={normalizeValue}
        onChange={(checked) => setModuleParam("channelmixerrgb", { [normalizeField]: checked })}
      />
    </>
  );

  return (
    <>
      <BauhausTabBar tabs={TAB_NAMES} value={tab} onChange={setTab} />

      {/* ===== CAT tab ===== */}
      {tab === "CAT" && (
        <>
          <BauhausCombo
            label="adaptation"
            options={ADAPTATION_LABELS}
            value={ADAPTATION_LABELS[adaptation] ?? ADAPTATION_LABELS[1]}
            onChange={(v) => {
              const idx = ADAPTATION_LABELS.indexOf(v);
              if (idx >= 0) setModuleParam("channelmixerrgb", { adaptation: idx });
            }}
          />

          <BauhausCombo
            label="illuminant"
            options={ILLUMINANT_LABELS}
            value={ILLUMINANT_LABELS[illuminant] ?? ILLUMINANT_LABELS[2]}
            onChange={(v) => {
              const idx = ILLUMINANT_LABELS.indexOf(v);
              if (idx >= 0) setModuleParam("channelmixerrgb", { illuminant: idx });
            }}
          />

          {showFluo && (
            <BauhausCombo
              label="fluorescent type"
              options={FLUO_LABELS}
              value={FLUO_LABELS[params.illum_fluo as number] ?? FLUO_LABELS[2]}
              onChange={(v) => {
                const idx = FLUO_LABELS.indexOf(v);
                if (idx >= 0) setModuleParam("channelmixerrgb", { illum_fluo: idx });
              }}
            />
          )}

          {showLed && (
            <BauhausCombo
              label="LED type"
              options={LED_LABELS}
              value={LED_LABELS[params.illum_led as number] ?? LED_LABELS[4]}
              onChange={(v) => {
                const idx = LED_LABELS.indexOf(v);
                if (idx >= 0) setModuleParam("channelmixerrgb", { illum_led: idx });
              }}
            />
          )}

          {showTemp && (
            <BauhausSlider
              label="temperature"
              value={localTemp}
              min={1667}
              max={25000}
              step={1}
              defaultValue={5003}
              format={(v) => `${Math.round(v)} K`}
              onChange={(v) => { setLocalTemp(v); throttledApply("temperature", v); }}
              onRelease={throttledRelease}
            />
          )}

          {showXY && (
            <>
              <BauhausSlider
                label="chromaticity x"
                value={params.x as number}
                min={0}
                max={0.8}
                step={0.001}
                defaultValue={0.333}
                format={(v) => v.toFixed(4)}
                onChange={(v) => setModuleParam("channelmixerrgb", { x: v })}
                onRelease={throttledRelease}
              />
              <BauhausSlider
                label="chromaticity y"
                value={params.y as number}
                min={0}
                max={0.8}
                step={0.001}
                defaultValue={0.333}
                format={(v) => v.toFixed(4)}
                onChange={(v) => setModuleParam("channelmixerrgb", { y: v })}
                onRelease={throttledRelease}
              />
            </>
          )}

          <BauhausSlider
            label="gamut compression"
            value={localGamut}
            min={0}
            max={12}
            step={0.01}
            defaultValue={1.0}
            format={(v) => v.toFixed(2)}
            onChange={(v) => { setLocalGamut(v); throttledApply("gamut", v); }}
            onRelease={throttledRelease}
          />

          <BauhausCheckbox
            label="clip negative RGB from gamut"
            checked={params.clip as boolean}
            onChange={(checked) => setModuleParam("channelmixerrgb", { clip: checked })}
          />
        </>
      )}

      {/* ===== R tab ===== */}
      {tab === "R" && renderMixerTab(
        localRed, setLocalRed, "red", "normalize_R", params.normalize_R as boolean,
        ["rgb(204,51,51)", "rgb(51,204,51)", "rgb(51,51,204)"],
      )}

      {/* ===== G tab ===== */}
      {tab === "G" && renderMixerTab(
        localGreen, setLocalGreen, "green", "normalize_G", params.normalize_G as boolean,
        ["rgb(204,51,51)", "rgb(51,204,51)", "rgb(51,51,204)"],
      )}

      {/* ===== B tab ===== */}
      {tab === "B" && renderMixerTab(
        localBlue, setLocalBlue, "blue", "normalize_B", params.normalize_B as boolean,
        ["rgb(204,51,51)", "rgb(51,204,51)", "rgb(51,51,204)"],
      )}

      {/* ===== Colorfulness tab ===== */}
      {tab === "colorfulness" && (
        <>
          {renderMixerTab(
            localSat, setLocalSat, "saturation", "normalize_sat", params.normalize_sat as boolean,
            ["rgb(204,51,51)", "rgb(51,204,51)", "rgb(51,51,204)"],
          )}
          <BauhausCombo
            label="saturation algorithm"
            options={VERSION_LABELS}
            value={VERSION_LABELS[params.version as number] ?? VERSION_LABELS[2]}
            onChange={(v) => {
              const idx = VERSION_LABELS.indexOf(v);
              if (idx >= 0) setModuleParam("channelmixerrgb", { version: idx });
            }}
          />
        </>
      )}

      {/* ===== Brightness tab ===== */}
      {tab === "brightness" && renderMixerTab(
        localLight, setLocalLight, "lightness", "normalize_light", params.normalize_light as boolean,
        ["rgb(204,51,51)", "rgb(51,204,51)", "rgb(51,51,204)"],
      )}

      {/* ===== Gray tab ===== */}
      {tab === "gray" && renderMixerTab(
        localGrey, setLocalGrey, "grey", "normalize_grey", params.normalize_grey as boolean,
      )}
    </>
  );
}
