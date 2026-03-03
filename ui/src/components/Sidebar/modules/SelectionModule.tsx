import CollapsibleModule from "../CollapsibleModule";
import ModuleButton from "../controls/ModuleButton";
import { useCatalogStore } from "../../../stores/catalogStore";

export default function SelectionModule() {
  const images = useCatalogStore((s) => s.images);
  const selectedIds = useCatalogStore((s) => s.selectedIds);
  const selectAll = useCatalogStore((s) => s.selectAll);
  const clearSelection = useCatalogStore((s) => s.clearSelection);
  const invertSelection = useCatalogStore((s) => s.invertSelection);
  const selectFilmRoll = useCatalogStore((s) => s.selectFilmRoll);
  const selectUntouched = useCatalogStore((s) => s.selectUntouched);

  const hasImages = images.length > 0;
  const hasSelection = selectedIds.size > 0;

  return (
    <CollapsibleModule title="selection" defaultOpen>
      <div className="module-button-grid">
        <ModuleButton label="select all" onClick={selectAll} disabled={!hasImages} />
        <ModuleButton label="select none" onClick={clearSelection} disabled={!hasSelection} />
        <ModuleButton label="invert selection" onClick={invertSelection} disabled={!hasImages} />
        <ModuleButton label="select film roll" onClick={selectFilmRoll} disabled={!hasSelection} />
        <ModuleButton label="select untouched" onClick={selectUntouched} disabled={!hasImages} />
      </div>
    </CollapsibleModule>
  );
}
