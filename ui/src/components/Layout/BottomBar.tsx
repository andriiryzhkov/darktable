import {
  Star,
  ZoomIn,
  ZoomOut,
  Ban,
  CircleOff,
} from "lucide-react";
import { useUIStore } from "../../stores/uiStore";

const COLORS = [
  { key: "red", var: "--colorlabel-red" },
  { key: "yellow", var: "--colorlabel-yellow" },
  { key: "green", var: "--colorlabel-green" },
  { key: "blue", var: "--colorlabel-blue" },
  { key: "purple", var: "--colorlabel-purple" },
];

export default function BottomBar() {
  const { thumbnailSize, setThumbnailSize, gridColumns } = useUIStore();


  const fewer = () => {
    if (gridColumns <= 1) return;
    const approxWidth = gridColumns * thumbnailSize;
    setThumbnailSize(Math.ceil(approxWidth / (gridColumns - 1)));
  };

  const more = () => {
    if (gridColumns <= 1) return;
    const approxWidth = gridColumns * thumbnailSize;
    setThumbnailSize(Math.floor(approxWidth / (gridColumns + 1)));
  };

  return (
    <div className="bottombar">
      {/* Left: rating + color labels */}
      <div className="flex items-center">
        <button className="bottombar-btn" title="Reject">
          <Ban size={12} />
        </button>

        {[1, 2, 3, 4, 5].map((n) => (
          <button key={n} className="bottombar-star" title={`Rate ${n} stars`}>
            <Star size={12} />
          </button>
        ))}

        <div className="bottombar-spacer" />

        {COLORS.map((c) => (
          <button key={c.key} className="bottombar-color" title={`Color label: ${c.key}`}>
            <div
              className="bottombar-color-dot"
              style={{ backgroundColor: `var(${c.var})` }}
            />
          </button>
        ))}

        <button className="bottombar-btn" title="Remove color label">
          <CircleOff size={12} />
        </button>
      </div>

      {/* Center: thumbs per row */}
      <div className="bottombar-pill">
        <span className="bottombar-pill-label">{gridColumns}</span>
        <button onClick={fewer} className="toolbar-icon-btn" title="Fewer thumbnails per row">
          <ZoomOut size={12} />
        </button>
        <button onClick={more} className="toolbar-icon-btn" title="More thumbnails per row">
          <ZoomIn size={12} />
        </button>
      </div>
    </div>
  );
}
