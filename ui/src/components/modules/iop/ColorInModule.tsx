import { useEffect } from "react";
import { useDevelopStore } from "../../../stores/developStore";
import BauhausCombo from "../../controls/BauhausCombo";
import type { ColorProfileEntry } from "../../../types/protocol";

const NORMALIZE_OPTIONS = [
  "off",
  "sRGB",
  "Adobe RGB (compatible)",
  "linear Rec709 RGB",
  "linear Rec2020 RGB",
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

export default function ColorInModule() {
  const colorinParams = useDevelopStore((s) => s.colorinParams);
  const setModuleParam = useDevelopStore((s) => s.setModuleParam);
  const fetchModuleParams = useDevelopStore((s) => s.fetchModuleParams);

  useEffect(() => {
    if (!colorinParams) {
      fetchModuleParams("colorin");
    }
  }, [colorinParams, fetchModuleParams]);

  if (!colorinParams) {
    return (
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        loading color profile params…
      </p>
    );
  }

  const inputProfiles = colorinParams.input_profiles;
  const workProfiles = colorinParams.work_profiles;

  const currentInputLabel =
    findProfileLabel(inputProfiles, colorinParams.type, colorinParams.filename)
    ?? colorinParams.input_profile_name;

  const currentWorkLabel =
    findProfileLabel(workProfiles, colorinParams.type_work, colorinParams.filename_work)
    ?? colorinParams.work_profile_name;

  return (
    <>
      <BauhausCombo
        label="input profile"
        options={inputProfiles.map((p) => p.name)}
        value={currentInputLabel}
        onChange={(label) => {
          const prof = inputProfiles.find((p) => p.name === label);
          if (prof) {
            setModuleParam("colorin", {
              type: prof.type,
              filename: prof.filename,
            });
          }
        }}
      />

      <BauhausCombo
        label="working profile"
        options={workProfiles.map((p) => p.name)}
        value={currentWorkLabel}
        onChange={(label) => {
          const prof = workProfiles.find((p) => p.name === label);
          if (prof) {
            setModuleParam("colorin", {
              type_work: prof.type,
              filename_work: prof.filename,
            });
          }
        }}
      />

      <BauhausCombo
        label="gamut clipping"
        options={NORMALIZE_OPTIONS}
        value={NORMALIZE_OPTIONS[colorinParams.normalize] ?? "off"}
        onChange={(label) => {
          const idx = NORMALIZE_OPTIONS.indexOf(label);
          if (idx >= 0) setModuleParam("colorin", { normalize: idx });
        }}
      />
    </>
  );
}
