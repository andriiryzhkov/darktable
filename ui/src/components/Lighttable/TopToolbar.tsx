import { useCatalogStore } from "../../stores/catalogStore";
import FilterBar from "./FilterBar";

export default function TopToolbar() {
  const { total, selectedIds } = useCatalogStore();

  return (
    <div className="top-toolbar">
      {/* Left: filters */}
      <FilterBar />

      {/* Right: selection info + future icons */}
      <div className="top-toolbar-right">
        <span className="top-toolbar-info">
          {selectedIds.size} of {total} selected
        </span>
      </div>
    </div>
  );
}
