import CollapsibleModule from "../CollapsibleModule";

export default function ScriptsModule() {
  return (
    <CollapsibleModule title="scripts" description="manage and run lua scripts">
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        lua scripts manager
      </p>
    </CollapsibleModule>
  );
}
