import CollapsibleModule from "../CollapsibleModule";

export default function TaggingModule() {
  return (
    <CollapsibleModule title="tagging">
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        attach / detach tags to selected images
      </p>
    </CollapsibleModule>
  );
}
