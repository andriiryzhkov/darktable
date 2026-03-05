import { useMemo } from "react";
import { Star, HelpCircle, Keyboard, Settings, Group, Ungroup } from "lucide-react";
import { useCatalogStore } from "../../stores/catalogStore";
import { useFilterStore } from "../../stores/filterStore";
import BauhausButton from "../controls/BauhausButton";
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
        <BauhausButton
          icon={grouping ? <Group size={14} /> : <Ungroup size={14} />}
          title={grouping ? "Grouped — click to show all" : "Ungrouped — click to collapse groups"}
          active={grouping}
          transparent
          onClick={toggleGrouping}
        />
        <BauhausButton icon={<Star size={14} />} title="Overlays" transparent />
        <BauhausButton icon={<HelpCircle size={14} />} title="Help" transparent />
        <BauhausButton icon={<Keyboard size={14} />} title="Keyboard shortcuts" transparent />
        <BauhausButton icon={<Settings size={14} />} title="Preferences" transparent />
      </div>
    </div>
  );
}
