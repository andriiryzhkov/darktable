import CollapsibleModule from "../CollapsibleModule";
import ModuleButton from "../controls/ModuleButton";
import ModuleCheckbox from "../controls/ModuleCheckbox";
import ModuleInput from "../controls/ModuleInput";
import ModuleSection from "../controls/ModuleSection";
import { useImportStore } from "../../../stores/importStore";

export default function ImportModule() {
  const openDialog = useImportStore((s) => s.openDialog);

  return (
    <CollapsibleModule title="import" defaultOpen>
      <div className="space-y-1">
        <div className="bauhaus-button-row">
          <ModuleButton label="add to library..." onClick={() => openDialog("inplace")} />
          <ModuleButton label="copy & import..." onClick={() => openDialog("copy")} />
        </div>
        <ModuleSection title="parameters">
          <div className="bauhaus-input-group">
            <ModuleCheckbox label="ignore EXIF rating" />
            <ModuleInput
              label="initial rating"
              type="integer"
              min={0}
              max={5}
              step={1}
              value={1}
            />
            <ModuleCheckbox label="apply metadata" checked />
            <ModuleInput
              label="metadata preset"
              type="select"
              options={["all rights reserved", "CC BY", "CC BY-SA", "public domain"]}
            />
            <ModuleInput label="title" value="" />
            <ModuleInput label="description" value="" />
            <ModuleInput label="creator" value="" />
            <ModuleInput label="publisher" value="" />
            <ModuleInput label="rights" value="all rights reserved" />
            <ModuleInput label="notes" value="" />
            <ModuleInput label="version name" value="" />
            <ModuleInput label="tag presets" type="select" options={["none"]} />
            <ModuleInput label="tags" value="" />
          </div>
        </ModuleSection>
      </div>
    </CollapsibleModule>
  );
}
