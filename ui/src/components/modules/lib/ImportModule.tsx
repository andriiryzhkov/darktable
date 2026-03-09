import LibModuleCard from "../LibModuleCard";
import BauhausButton from "../../controls/BauhausButton";
import BauhausCheckbox from "../../controls/BauhausCheckbox";
import BauhausInput from "../../controls/BauhausInput";
import BauhausCollapsible from "../../controls/BauhausCollapsible";
import { useImportStore } from "../../../stores/importStore";

export default function ImportModule() {
  const openDialog = useImportStore((s) => s.openDialog);

  return (
    <LibModuleCard title="import" description="add images to the library or copy and import from external locations">
      <div className="space-y-1">
        <div className="bauhaus-button-row">
          <BauhausButton label="add to library..." onClick={() => openDialog("inplace")} />
          <BauhausButton label="copy & import..." onClick={() => openDialog("copy")} />
        </div>
        <BauhausCollapsible title="parameters">
          <div className="bauhaus-input-group">
            <BauhausCheckbox label="ignore EXIF rating" />
            <BauhausInput
              label="initial rating"
              type="integer"
              min={0}
              max={5}
              step={1}
              value={1}
            />
            <BauhausCheckbox label="apply metadata" checked />
            <BauhausInput
              label="metadata preset"
              type="select"
              options={["all rights reserved", "CC BY", "CC BY-SA", "public domain"]}
            />
            <BauhausInput label="title" value="" />
            <BauhausInput label="description" value="" />
            <BauhausInput label="creator" value="" />
            <BauhausInput label="publisher" value="" />
            <BauhausInput label="rights" value="all rights reserved" />
            <BauhausInput label="notes" value="" />
            <BauhausInput label="version name" value="" />
            <BauhausInput label="tag presets" type="select" options={["none"]} />
            <BauhausInput label="tags" value="" />
          </div>
        </BauhausCollapsible>
      </div>
    </LibModuleCard>
  );
}
