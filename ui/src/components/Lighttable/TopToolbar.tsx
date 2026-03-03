import { ArrowUpDown, Filter } from "lucide-react";
import { useCatalogStore } from "../../stores/catalogStore";

export default function TopToolbar() {
  const { total, selectedIds } = useCatalogStore();

  return (
    <div
      className="flex items-center justify-between px-3 shrink-0"
      style={{
        height: 30,
        backgroundColor: "var(--plugin-bg-color)",
        borderBottom: "1px solid var(--border-color)",
      }}
    >
      {/* Left: collection filter + rating/color filters */}
      <div className="flex items-center gap-2">
        <select
          className="dt-select"
          style={{ width: 120 }}
          defaultValue="all"
        >
          <option value="all">All images</option>
        </select>
        <button
          className="flex items-center justify-center"
          style={{
            width: 22,
            height: 22,
            color: "var(--plugin-label-color)",
          }}
          title="Filter"
        >
          <Filter size={12} />
        </button>
      </div>

      {/* Center: sort */}
      <div className="flex items-center gap-2">
        <select
          className="dt-select"
          style={{ width: 120 }}
          defaultValue="capture_time"
        >
          <option value="capture_time">capture time</option>
          <option value="filename">filename</option>
          <option value="import_time">import time</option>
        </select>
        <button
          className="flex items-center justify-center"
          style={{
            width: 22,
            height: 22,
            color: "var(--plugin-label-color)",
          }}
          title="Sort direction"
        >
          <ArrowUpDown size={12} />
        </button>
      </div>

      {/* Right: selection info */}
      <span
        className="text-xs"
        style={{ color: "var(--plugin-label-color)" }}
      >
        {selectedIds.size} of {total} selected
      </span>
    </div>
  );
}
