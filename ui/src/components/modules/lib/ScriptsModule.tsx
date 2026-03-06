import LibModuleCard from "../LibModuleCard";

export default function ScriptsModule() {
  return (
    <LibModuleCard title="scripts" description="manage and run lua scripts">
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        lua scripts manager
      </p>
    </LibModuleCard>
  );
}
