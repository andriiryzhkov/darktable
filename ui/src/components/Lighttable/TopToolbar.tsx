import { ArrowUpDown, Filter } from "lucide-react";
import { useCatalogStore } from "../../stores/catalogStore";

export default function TopToolbar() {
  const { total, selectedIds } = useCatalogStore();

  return (
    <div className="top-toolbar">
      {/* Left: collection filter */}
      <div className="flex items-center gap-2">
        <select className="dt-select top-toolbar-select" defaultValue="all">
          <option value="all">All images</option>
        </select>
        <button className="toolbar-icon-btn" title="Filter">
          <Filter size={11} />
        </button>
      </div>

      {/* Center: sort */}
      <div className="flex items-center gap-2">
        <select className="dt-select top-toolbar-select" defaultValue="capture_time">
          <option value="capture_time">capture time</option>
          <option value="filename">filename</option>
          <option value="import_time">import time</option>
        </select>
        <button className="toolbar-icon-btn" title="Sort direction">
          <ArrowUpDown size={11} />
        </button>
      </div>

      {/* Right: selection info */}
      <span className="top-toolbar-info">
        {selectedIds.size} of {total} selected
      </span>
    </div>
  );
}
