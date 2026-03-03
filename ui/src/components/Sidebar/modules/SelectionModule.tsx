import CollapsibleModule from "../CollapsibleModule";

export default function SelectionModule() {
  return (
    <CollapsibleModule title="selection">
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        select all, select none, invert selection
      </p>
    </CollapsibleModule>
  );
}
