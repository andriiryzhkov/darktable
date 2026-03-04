import { useMemo } from "react";
import { Star, HelpCircle, Keyboard, Settings } from "lucide-react";
import { useCatalogStore } from "../../stores/catalogStore";
import { useFilterStore } from "../../stores/filterStore";
import FilterBar from "./FilterBar";

export default function TopToolbar() {
  const { images, selectedIds } = useCatalogStore();
  const grouping = useFilterStore((s) => s.grouping);
  const toggleGrouping = useFilterStore((s) => s.toggleGrouping);
  const displayedCount = useMemo(
    () => grouping ? images.filter((img) => img.id === (img.group_id ?? img.id)).length : images.length,
    [images, grouping],
  );

  return (
    <div className="top-toolbar">
      {/* Left: filters */}
      <FilterBar />

      {/* Center: selection info */}
      <span className="top-toolbar-info">
        {selectedIds.size} of {displayedCount} selected
      </span>

      {/* Right: action buttons */}
      <div className="top-toolbar-right">
        <button
          className="toolbar-icon-btn"
          title={grouping ? "Grouped — click to show all" : "Ungrouped — click to collapse groups"}
          data-active={grouping}
          onClick={toggleGrouping}
        >
          <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinejoin="round">
            <rect x="9" y="9" width="13" height="13" rx="2" fill={grouping ? "currentColor" : "none"} />
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
          </svg>
        </button>
        <button className="toolbar-icon-btn" title="Overlays">
          <Star size={14} />
        </button>
        <button className="toolbar-icon-btn" title="Help">
          <HelpCircle size={14} />
        </button>
        <button className="toolbar-icon-btn" title="Keyboard shortcuts">
          <Keyboard size={14} />
        </button>
        <button className="toolbar-icon-btn" title="Preferences">
          <Settings size={14} />
        </button>
      </div>
    </div>
  );
}
