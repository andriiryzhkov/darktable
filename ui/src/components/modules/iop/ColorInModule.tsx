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
  const params = useDevelopStore((s) => s.genericParams["colorin"]);
  const setModuleParam = useDevelopStore((s) => s.setModuleParam);
  const fetchGenericParams = useDevelopStore((s) => s.fetchGenericParams);

  useEffect(() => {
    if (!params) {
      fetchGenericParams("colorin");
    }
  }, [params, fetchGenericParams]);

  if (!params) {
    return (
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        loading color profile params…
      </p>
    );
  }

  const inputProfiles = params.input_profiles as ColorProfileEntry[];
  const workProfiles = params.work_profiles as ColorProfileEntry[];

  const currentInputLabel =
    findProfileLabel(inputProfiles, params.type as number, params.filename as string)
    ?? (params.input_profile_name as string);

  const currentWorkLabel =
    findProfileLabel(workProfiles, params.type_work as number, params.filename_work as string)
    ?? (params.work_profile_name as string);

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
        value={NORMALIZE_OPTIONS[params.normalize as number] ?? "off"}
        onChange={(label) => {
          const idx = NORMALIZE_OPTIONS.indexOf(label);
          if (idx >= 0) setModuleParam("colorin", { normalize: idx });
        }}
      />
    </>
  );
}
