import CollapsibleModule from "../CollapsibleModule";

export default function ImportModule() {
  return (
    <CollapsibleModule title="import">
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        import images from folder or camera
      </p>
    </CollapsibleModule>
  );
}
