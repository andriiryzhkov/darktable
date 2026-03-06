import CollapsibleModule from "../CollapsibleModule";

export default function StylesModule() {
  return (
    <CollapsibleModule title="styles" description="apply styles to the currently selected images or manage your styles">
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        apply or create editing styles
      </p>
    </CollapsibleModule>
  );
}
