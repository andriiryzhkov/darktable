import { useState, useEffect } from "react";
import { useDevelopStore } from "../../../stores/developStore";
import { useThrottledParam } from "../../../hooks/useThrottledParam";
import BauhausSlider from "../../controls/BauhausSlider";
import BauhausCombo from "../../controls/BauhausCombo";
import BauhausCollapsible from "../../controls/BauhausCollapsible";

const RAD_2_DEG = 180 / Math.PI;

export default function SigmoidModule() {
  const params = useDevelopStore((s) => s.genericParams["sigmoid"]);
  const setModuleParam = useDevelopStore((s) => s.setModuleParam);
  const fetchGenericParams = useDevelopStore((s) => s.fetchGenericParams);

  const [localContrast, setLocalContrast] = useState(1.5);
  const [localSkew, setLocalSkew] = useState(0);
  const [localHue, setLocalHue] = useState(100);
  const [localWhite, setLocalWhite] = useState(100);
  const [localBlack, setLocalBlack] = useState(0.0152);
  const [localRedInset, setLocalRedInset] = useState(0);
  const [localRedRotation, setLocalRedRotation] = useState(0);
  const [localGreenInset, setLocalGreenInset] = useState(0);
  const [localGreenRotation, setLocalGreenRotation] = useState(0);
  const [localBlueInset, setLocalBlueInset] = useState(0);
  const [localBlueRotation, setLocalBlueRotation] = useState(0);
  const [localPurity, setLocalPurity] = useState(0);

  const { apply: throttledApply, release: throttledRelease, draggingRef } = useThrottledParam("sigmoid");

  useEffect(() => {
    if (!params) {
      fetchGenericParams("sigmoid");
    }
  }, [params, fetchGenericParams]);

  useEffect(() => {
    if (params && !draggingRef.current) {
      setLocalContrast(params.middle_grey_contrast as number);
      setLocalSkew(params.contrast_skewness as number);
      setLocalHue(params.hue_preservation as number);
      setLocalWhite(params.display_white_target as number);
      setLocalBlack(params.display_black_target as number);
      setLocalRedInset(params.red_inset as number);
      setLocalRedRotation(params.red_rotation as number);
      setLocalGreenInset(params.green_inset as number);
      setLocalGreenRotation(params.green_rotation as number);
      setLocalBlueInset(params.blue_inset as number);
      setLocalBlueRotation(params.blue_rotation as number);
      setLocalPurity(params.purity as number);
    }
  }, [params, draggingRef]);

  if (!params) {
    return (
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        loading sigmoid params…
      </p>
    );
  }

  const isPerChannel = params.color_processing === 0;

  return (
    <>
      <BauhausSlider
        label="contrast"
        value={localContrast}
        min={0.1}
        max={10}
        step={0.001}
        defaultValue={1.5}
        format={(v) => v.toFixed(3)}
        onChange={(v) => { setLocalContrast(v); throttledApply("middle_grey_contrast", v); }}
        onRelease={throttledRelease}
      />

      <BauhausSlider
        label="skew"
        value={localSkew}
        min={-1}
        max={1}
        step={0.001}
        defaultValue={0}
        origin={0}
        format={(v) => v.toFixed(3)}
        onChange={(v) => { setLocalSkew(v); throttledApply("contrast_skewness", v); }}
        onRelease={throttledRelease}
      />

      <BauhausCombo
        label="color processing"
        options={["per channel", "RGB ratio"]}
        value={isPerChannel ? "per channel" : "RGB ratio"}
        onChange={(v) =>
          setModuleParam("sigmoid", { color_processing: v === "per channel" ? 0 : 1 })
        }
      />

      {isPerChannel && (
        <BauhausSlider
          label="preserve hue"
          value={localHue}
          min={0}
          max={100}
          step={0.01}
          defaultValue={100}
          format={(v) => `${v.toFixed(2)}%`}
          onChange={(v) => { setLocalHue(v); throttledApply("hue_preservation", v); }}
          onRelease={throttledRelease}
        />
      )}

      <BauhausCollapsible title="display luminance">
        <BauhausSlider
          label="target black"
          value={localBlack}
          min={0}
          max={15}
          step={0.0001}
          defaultValue={0.0152}
          format={(v) => `${v.toFixed(4)}%`}
          onChange={(v) => { setLocalBlack(v); throttledApply("display_black_target", v); }}
          onRelease={throttledRelease}
        />

        <BauhausSlider
          label="target white"
          value={localWhite}
          min={20}
          max={1600}
          step={0.1}
          defaultValue={100}
          format={(v) => `${v.toFixed(1)}%`}
          onChange={(v) => { setLocalWhite(v); throttledApply("display_white_target", v); }}
          onRelease={throttledRelease}
        />
      </BauhausCollapsible>

      {isPerChannel && (
        <BauhausCollapsible title="primaries">
          <BauhausCombo
            label="base primaries"
            options={["working profile", "Rec2020", "Display P3", "Adobe RGB", "sRGB"]}
            value={["working profile", "Rec2020", "Display P3", "Adobe RGB", "sRGB"][params.base_primaries as number] ?? "working profile"}
            onChange={(v) => {
              const idx = ["working profile", "Rec2020", "Display P3", "Adobe RGB", "sRGB"].indexOf(v);
              setModuleParam("sigmoid", { base_primaries: idx >= 0 ? idx : 0 });
            }}
          />

          <BauhausSlider
            label="red attenuation"
            value={localRedInset * 100}
            min={0}
            max={99}
            step={0.1}
            defaultValue={0}
            color="rgb(204,51,51)"
            format={(v) => `${v.toFixed(1)}%`}
            onChange={(v) => { const raw = v / 100; setLocalRedInset(raw); throttledApply("red_inset", raw); }}
            onRelease={throttledRelease}
          />
          <BauhausSlider
            label="red rotation"
            value={localRedRotation * RAD_2_DEG}
            min={-0.4 * RAD_2_DEG}
            max={0.4 * RAD_2_DEG}
            step={0.1}
            defaultValue={0}
            origin={0}
            color="rgb(204,51,51)"
            format={(v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}°`}
            onChange={(v) => { const raw = v / RAD_2_DEG; setLocalRedRotation(raw); throttledApply("red_rotation", raw); }}
            onRelease={throttledRelease}
          />

          <BauhausSlider
            label="green attenuation"
            value={localGreenInset * 100}
            min={0}
            max={99}
            step={0.1}
            defaultValue={0}
            color="rgb(51,204,51)"
            format={(v) => `${v.toFixed(1)}%`}
            onChange={(v) => { const raw = v / 100; setLocalGreenInset(raw); throttledApply("green_inset", raw); }}
            onRelease={throttledRelease}
          />
          <BauhausSlider
            label="green rotation"
            value={localGreenRotation * RAD_2_DEG}
            min={-0.4 * RAD_2_DEG}
            max={0.4 * RAD_2_DEG}
            step={0.1}
            defaultValue={0}
            origin={0}
            color="rgb(51,204,51)"
            format={(v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}°`}
            onChange={(v) => { const raw = v / RAD_2_DEG; setLocalGreenRotation(raw); throttledApply("green_rotation", raw); }}
            onRelease={throttledRelease}
          />

          <BauhausSlider
            label="blue attenuation"
            value={localBlueInset * 100}
            min={0}
            max={99}
            step={0.1}
            defaultValue={0}
            color="rgb(51,51,204)"
            format={(v) => `${v.toFixed(1)}%`}
            onChange={(v) => { const raw = v / 100; setLocalBlueInset(raw); throttledApply("blue_inset", raw); }}
            onRelease={throttledRelease}
          />
          <BauhausSlider
            label="blue rotation"
            value={localBlueRotation * RAD_2_DEG}
            min={-0.4 * RAD_2_DEG}
            max={0.4 * RAD_2_DEG}
            step={0.1}
            defaultValue={0}
            origin={0}
            color="rgb(51,51,204)"
            format={(v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}°`}
            onChange={(v) => { const raw = v / RAD_2_DEG; setLocalBlueRotation(raw); throttledApply("blue_rotation", raw); }}
            onRelease={throttledRelease}
          />

          <BauhausSlider
            label="recover purity"
            value={localPurity * 100}
            min={0}
            max={100}
            step={1}
            defaultValue={0}
            format={(v) => `${v.toFixed(0)}%`}
            onChange={(v) => { const raw = v / 100; setLocalPurity(raw); throttledApply("purity", raw); }}
            onRelease={throttledRelease}
          />
        </BauhausCollapsible>
      )}
    </>
  );
}
