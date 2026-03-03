import { useState } from "react";
import CollapsibleModule from "../CollapsibleModule";
import ModuleCombo from "../controls/ModuleCombo";
import ModuleRow from "../controls/ModuleRow";
import ModuleSlider from "../controls/ModuleSlider";
import ModuleTextInput from "../controls/ModuleTextInput";

export default function ExportModule() {
  const [quality, setQuality] = useState(95);

  return (
    <CollapsibleModule title="export" defaultOpen>
      <div className="space-y-1">
        <ModuleCombo label="target storage" options={["file on disk"]} />
        <ModuleRow label="">
          <ModuleTextInput value="Export/$(FILE_NAME)$(VERSION)" />
        </ModuleRow>
        <ModuleCombo label="on conflict" options={["create unique filename"]} />
        <ModuleCombo
          label="file format"
          options={["JPEG (8 bit)", "PNG (8 bit)", "TIFF (16 bit)", "EXR (32 bit)"]}
        />
        <ModuleSlider
          label="quality"
          min={1}
          max={100}
          step={1}
          defaultValue={95}
          value={quality}
          onChange={setQuality}
          format={(v) => String(Math.round(v))}
        />
        <ModuleCombo label="chroma subsampling" options={["auto"]} />
        <ModuleCombo label="set size" options={["by scale (for file)"]} />
        <ModuleRow label="">
          <ModuleTextInput value="0.85" />
        </ModuleRow>
        <ModuleCombo label="allow upscaling" options={["no", "yes"]} />
        <ModuleCombo label="high quality resampling" options={["yes", "no"]} />
        <ModuleCombo label="store masks" options={["no", "yes"]} />
        <ModuleCombo label="profile" options={["sRGB", "AdobeRGB", "ProPhoto RGB"]} />
        <ModuleCombo
          label="intent"
          options={["perceptual", "relative colorimetric", "saturation"]}
        />
        <ModuleCombo label="style" options={["none"]} />
        <div className="mt-2">
          <button
            className="w-full py-1.5 text-xs rounded"
            style={{
              backgroundColor: "var(--button-bg)",
              color: "var(--button-fg)",
              border: "1px solid var(--button-border)",
            }}
          >
            Export
          </button>
        </div>
      </div>
    </CollapsibleModule>
  );
}
