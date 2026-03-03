import CollapsibleModule from "../CollapsibleModule";

export default function EditMetadataModule() {
  return (
    <CollapsibleModule title="edit metadata">
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        edit image metadata fields
      </p>
    </CollapsibleModule>
  );
}
