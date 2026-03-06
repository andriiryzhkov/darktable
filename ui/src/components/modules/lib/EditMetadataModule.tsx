import CollapsibleModule from "../CollapsibleModule";

export default function EditMetadataModule() {
  return (
    <CollapsibleModule title="edit metadata" description="modify text metadata fields of the currently selected images">
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        edit image metadata fields
      </p>
    </CollapsibleModule>
  );
}
