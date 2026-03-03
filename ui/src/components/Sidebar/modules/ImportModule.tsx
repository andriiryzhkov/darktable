import CollapsibleModule from "../CollapsibleModule";
import ModuleCheckbox from "../controls/ModuleCheckbox";
import ModuleInput from "../controls/ModuleInput";
import ModuleSection from "../controls/ModuleSection";

export default function ImportModule() {
  return (
    <CollapsibleModule title="import" defaultOpen>
      <div className="space-y-1">
        <div className="flex gap-2 mb-2">
          <button
            className="flex-1 py-1.5 text-xs rounded"
            style={{
              backgroundColor: "var(--button-bg)",
              color: "var(--button-fg)",
              border: "1px solid var(--button-border)",
            }}
          >
            add to library...
          </button>
          <button
            className="flex-1 py-1.5 text-xs rounded"
            style={{
              backgroundColor: "var(--button-bg)",
              color: "var(--button-fg)",
              border: "1px solid var(--button-border)",
            }}
          >
            copy &amp; import...
          </button>
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
