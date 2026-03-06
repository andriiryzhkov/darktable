import LibModuleCard from "../LibModuleCard";

export default function TaggingModule() {
  return (
    <LibModuleCard title="tagging" description="add or remove keywords for the currently selected images">
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        attach / detach tags to selected images
      </p>
    </LibModuleCard>
  );
}
