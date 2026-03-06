import { useMemo } from "react";
import { Star, HelpCircle, Keyboard, Settings, Group, Ungroup } from "lucide-react";
import { useCatalogStore } from "../../stores/catalogStore";
import { useFilterStore } from "../../stores/filterStore";
import BauhausButton from "../controls/BauhausButton";
import BauhausTooltip from "../controls/BauhausTooltip";
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
        <BauhausTooltip content={grouping ? "expand grouped images" : "collapse grouped images"}>
          <BauhausButton
            icon={grouping ? <Group size={14} /> : <Ungroup size={14} />}
            active={grouping}
            transparent
            onClick={toggleGrouping}
          />
        </BauhausTooltip>
        <BauhausTooltip content="click to change the type of overlays shown on thumbnails">
          <BauhausButton icon={<Star size={14} />} transparent />
        </BauhausTooltip>
        <BauhausTooltip content="enable this, then click on a control element to see its online help">
          <BauhausButton icon={<HelpCircle size={14} />} transparent />
        </BauhausTooltip>
        <BauhausTooltip content={"define keyboard shortcuts for on-screen controls\nctrl+click to switch off overwrite confirmations\n\nafter activating:\n\n- hover over a control and press a keystroke combination\n  to define a shortcut for the control\n- type an existing combination to delete that mapping\n\nclick on a control, module or screen area to open the\ndialog for more detailed configuration\n\nright-click to exit mapping mode"}>
          <BauhausButton icon={<Keyboard size={14} />} transparent />
        </BauhausTooltip>
        <BauhausTooltip content="show global preferences">
          <BauhausButton icon={<Settings size={14} />} transparent />
        </BauhausTooltip>
      </div>
    </div>
  );
}
