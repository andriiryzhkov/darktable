import CollapsibleModule from "../CollapsibleModule";

export default function CollectionFiltersModule() {
  return (
    <CollapsibleModule title="collection filters" defaultOpen>
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        filter images by metadata, rating, color label
      </p>
    </CollapsibleModule>
  );
}
