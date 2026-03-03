import CollapsibleModule from "../CollapsibleModule";

export default function StylesModule() {
  return (
    <CollapsibleModule title="styles">
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        apply or create editing styles
      </p>
    </CollapsibleModule>
  );
}
