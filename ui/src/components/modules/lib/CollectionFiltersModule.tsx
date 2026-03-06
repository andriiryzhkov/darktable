import CollapsibleModule from "../CollapsibleModule";

export default function CollectionFiltersModule() {
  return (
    <CollapsibleModule title="collection filters" description="refine the set of images to display or edit - filters can be pinned to the top toolbar, where they will also be visible in the darkroom" defaultOpen>
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        filter images by metadata, rating, color label
      </p>
    </CollapsibleModule>
  );
}
