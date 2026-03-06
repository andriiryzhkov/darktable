import LibModuleCard from "../LibModuleCard";

export default function EditMetadataModule() {
  return (
    <LibModuleCard title="edit metadata" description="modify text metadata fields of the currently selected images">
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        edit image metadata fields
      </p>
    </LibModuleCard>
  );
}
