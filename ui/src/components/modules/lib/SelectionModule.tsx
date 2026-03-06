import LibModuleCard from "../LibModuleCard";
import BauhausButton from "../../controls/BauhausButton";
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
    <LibModuleCard title="selection" description="modify which of the displayed images are selected" >
      <div className="module-button-grid">
        <BauhausButton label="select all" onClick={selectAll} disabled={!hasImages} />
        <BauhausButton label="select none" onClick={clearSelection} disabled={!hasSelection} />
        <BauhausButton label="invert selection" onClick={invertSelection} disabled={!hasImages} />
        <BauhausButton label="select film roll" onClick={selectFilmRoll} disabled={!hasSelection} />
        <BauhausButton label="select untouched" onClick={selectUntouched} disabled={!hasImages} />
      </div>
    </LibModuleCard>
  );
}
