import { useState } from "react";
import CollapsibleModule from "../CollapsibleModule";
import BauhausButton from "../../controls/BauhausButton";
import BauhausCombo from "../../controls/BauhausCombo";
import BauhausRow from "../../controls/BauhausRow";
import BauhausSlider from "../../controls/BauhausSlider";
import BauhausTextInput from "../../controls/BauhausTextInput";

export default function ExportModule() {
  const [quality, setQuality] = useState(95);

  return (
    <CollapsibleModule title="export" defaultOpen>
      <div className="space-y-1">
        <BauhausCombo label="target storage" options={["file on disk"]} />
        <BauhausRow label="">
          <BauhausTextInput value="Export/$(FILE_NAME)$(VERSION)" />
        </BauhausRow>
        <BauhausCombo label="on conflict" options={["create unique filename"]} />
        <BauhausCombo
          label="file format"
          options={["JPEG (8 bit)", "PNG (8 bit)", "TIFF (16 bit)", "EXR (32 bit)"]}
        />
        <BauhausSlider
          label="quality"
          min={1}
          max={100}
          step={1}
          defaultValue={95}
          value={quality}
          onChange={setQuality}
          format={(v) => String(Math.round(v))}
        />
        <BauhausCombo label="chroma subsampling" options={["auto"]} />
        <BauhausCombo label="set size" options={["by scale (for file)"]} />
        <BauhausRow label="">
          <BauhausTextInput value="0.85" />
        </BauhausRow>
        <BauhausCombo label="allow upscaling" options={["no", "yes"]} />
        <BauhausCombo label="high quality resampling" options={["yes", "no"]} />
        <BauhausCombo label="store masks" options={["no", "yes"]} />
        <BauhausCombo label="profile" options={["sRGB", "AdobeRGB", "ProPhoto RGB"]} />
        <BauhausCombo
          label="intent"
          options={["perceptual", "relative colorimetric", "saturation"]}
        />
        <BauhausCombo label="style" options={["none"]} />
        <BauhausButton label="export" />
      </div>
    </CollapsibleModule>
  );
}
