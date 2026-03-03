import CollapsibleModule from "../CollapsibleModule";
import ModuleCombo from "../controls/ModuleCombo";
import ModuleRow from "../controls/ModuleRow";
import ModuleSection from "../controls/ModuleSection";
import ModuleSlider from "../controls/ModuleSlider";
import ModuleTextInput from "../controls/ModuleTextInput";

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
          <ModuleCombo label="ignore EXIF rating" options={["no", "yes"]} />
          <ModuleSlider
            label="initial rating"
            min={0}
            max={5}
            step={1}
            defaultValue={1}
            value={1}
            format={(v) => String(Math.round(v))}
          />
          <ModuleCombo label="apply metadata" options={["yes", "no"]} />
          <ModuleCombo
            label="metadata preset"
            options={["all rights reserved", "CC BY", "CC BY-SA", "public domain"]}
          />
          <ModuleRow label="title">
            <ModuleTextInput value="" />
          </ModuleRow>
          <ModuleRow label="description">
            <ModuleTextInput value="" />
          </ModuleRow>
          <ModuleRow label="creator">
            <ModuleTextInput value="" />
          </ModuleRow>
          <ModuleRow label="publisher">
            <ModuleTextInput value="" />
          </ModuleRow>
          <ModuleRow label="rights">
            <ModuleTextInput value="all rights reserved" />
          </ModuleRow>
          <ModuleRow label="notes">
            <ModuleTextInput value="" />
          </ModuleRow>
          <ModuleRow label="version name">
            <ModuleTextInput value="" />
          </ModuleRow>
          <ModuleCombo label="tag presets" options={["none"]} />
          <ModuleRow label="tags">
            <ModuleTextInput value="" />
          </ModuleRow>
        </ModuleSection>
      </div>
    </CollapsibleModule>
  );
}
