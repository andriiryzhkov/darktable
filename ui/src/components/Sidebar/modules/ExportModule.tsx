import { useState } from "react";
import CollapsibleModule from "../CollapsibleModule";
import ModuleRow from "../controls/ModuleRow";
import ModuleSelect from "../controls/ModuleSelect";
import ModuleSlider from "../controls/ModuleSlider";
import ModuleTextInput from "../controls/ModuleTextInput";

export default function ExportModule() {
  const [quality, setQuality] = useState(97);

  return (
    <CollapsibleModule title="export" defaultOpen>
      <div className="space-y-1">
        <ModuleRow label="target storage">
          <ModuleSelect options={["file on disk"]} />
        </ModuleRow>
        <ModuleRow label="">
          <ModuleTextInput value="Export/$(FILE_NAME)$(VERSION)" />
        </ModuleRow>
        <ModuleRow label="on conflict">
          <ModuleSelect options={["create unique filename"]} />
        </ModuleRow>
        <ModuleRow label="file format">
          <ModuleSelect options={["JPEG (8 bit)", "PNG (8 bit)", "TIFF (16 bit)", "EXR (32 bit)"]} />
        </ModuleRow>
        <ModuleSlider
          label="quality"
          min={1}
          max={100}
          step={1}
          value={quality}
          onChange={setQuality}
          format={(v) => String(Math.round(v))}
        />
        <ModuleRow label="chroma subsampling">
          <ModuleSelect options={["auto"]} />
        </ModuleRow>
        <ModuleRow label="set size">
          <ModuleSelect options={["by scale (for file)"]} />
        </ModuleRow>
        <ModuleRow label="">
          <ModuleTextInput value="0.85" />
        </ModuleRow>
        <ModuleRow label="allow upscaling">
          <ModuleSelect options={["no", "yes"]} />
        </ModuleRow>
        <ModuleRow label="high quality resampling">
          <ModuleSelect options={["yes", "no"]} />
        </ModuleRow>
        <ModuleRow label="store masks">
          <ModuleSelect options={["no", "yes"]} />
        </ModuleRow>
        <ModuleRow label="profile">
          <ModuleSelect options={["sRGB", "AdobeRGB", "ProPhoto RGB"]} />
        </ModuleRow>
        <ModuleRow label="intent">
          <ModuleSelect options={["perceptual", "relative colorimetric", "saturation"]} />
        </ModuleRow>
        <ModuleRow label="style">
          <ModuleSelect options={["none"]} />
        </ModuleRow>
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
