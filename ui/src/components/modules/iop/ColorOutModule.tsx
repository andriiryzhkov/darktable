import { useEffect } from "react";
import { useDevelopStore } from "../../../stores/developStore";
import BauhausCombo from "../../controls/BauhausCombo";
import type { ColorProfileEntry } from "../../../types/protocol";

const INTENT_OPTIONS = [
  "perceptual",
  "relative colorimetric",
  "saturation",
  "absolute colorimetric",
];

function findProfileLabel(
  profiles: ColorProfileEntry[],
  type: number,
  filename: string,
): string | undefined {
  return profiles.find(
    (p) => p.type === type && p.filename === filename,
  )?.name;
}

export default function ColorOutModule() {
  const coloroutParams = useDevelopStore((s) => s.coloroutParams);
  const setModuleParam = useDevelopStore((s) => s.setModuleParam);
  const fetchModuleParams = useDevelopStore((s) => s.fetchModuleParams);

  useEffect(() => {
    if (!coloroutParams) {
      fetchModuleParams("colorout");
    }
  }, [coloroutParams, fetchModuleParams]);

  if (!coloroutParams) {
    return (
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        loading output color profile params…
      </p>
    );
  }

  const outputProfiles = coloroutParams.output_profiles;

  const currentLabel =
    findProfileLabel(outputProfiles, coloroutParams.type, coloroutParams.filename)
    ?? coloroutParams.output_profile_name;

  return (
    <>
      <BauhausCombo
        label="output intent"
        options={INTENT_OPTIONS}
        value={INTENT_OPTIONS[coloroutParams.intent] ?? "perceptual"}
        onChange={(label) => {
          const idx = INTENT_OPTIONS.indexOf(label);
          if (idx >= 0) setModuleParam("colorout", { intent: idx });
        }}
      />

      <BauhausCombo
        label="export profile"
        options={outputProfiles.map((p) => p.name)}
        value={currentLabel}
        onChange={(label) => {
          const prof = outputProfiles.find((p) => p.name === label);
          if (prof) {
            setModuleParam("colorout", {
              type: prof.type,
              filename: prof.filename,
            });
          }
        }}
      />
    </>
  );
}
