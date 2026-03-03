import {
  Star,
  ZoomIn,
  ZoomOut,
  Ban,
  CircleOff,
} from "lucide-react";
import { useUIStore } from "../../stores/uiStore";
import { useCatalogStore } from "../../stores/catalogStore";

const COLORS = [
  { key: "red", var: "--colorlabel-red" },
  { key: "yellow", var: "--colorlabel-yellow" },
  { key: "green", var: "--colorlabel-green" },
  { key: "blue", var: "--colorlabel-blue" },
  { key: "purple", var: "--colorlabel-purple" },
];

export default function BottomBar() {
  const { thumbnailSize, setThumbnailSize, gridColumns } = useUIStore();
  const total = useCatalogStore((s) => s.total);

  const fewer = () => {
    // fewer per row = larger thumbs
    if (gridColumns <= 1) return;
    const approxWidth = gridColumns * thumbnailSize;
    setThumbnailSize(Math.ceil(approxWidth / (gridColumns - 1)));
  };

  const more = () => {
    // more per row = smaller thumbs
    if (gridColumns <= 1) return;
    const approxWidth = gridColumns * thumbnailSize;
    setThumbnailSize(Math.floor(approxWidth / (gridColumns + 1)));
  };

  return (
    <div
      className="flex items-center justify-between px-3 shrink-0"
      style={{
        height: 32,
        backgroundColor: "var(--plugin-bg-color)",
      }}
    >
      {/* Left: rating + color labels */}
      <div className="flex items-center">
        {/* Reject */}
        <button
          className="flex items-center justify-center"
          style={{
            width: 22,
            height: 22,
            color: "var(--plugin-label-color)",
          }}
          title="Reject"
        >
          <Ban size={13} />
        </button>

        {/* Stars */}
        {[1, 2, 3, 4, 5].map((n) => (
          <button
            key={n}
            className="flex items-center justify-center"
            style={{
              width: 26,
              height: 26,
              color: "var(--plugin-label-color)",
            }}
            title={`Rate ${n} stars`}
          >
            <Star size={15} />
          </button>
        ))}

        <div style={{ width: 8 }} />

        {/* Color labels */}
        {COLORS.map((c) => (
          <button
            key={c.key}
            className="flex items-center justify-center"
            style={{ width: 26, height: 26 }}
            title={`Color label: ${c.key}`}
          >
            <div
              className="rounded-full"
              style={{
                width: 12,
                height: 12,
                backgroundColor: `var(${c.var})`,
              }}
            />
          </button>
        ))}

        {/* Remove color label */}
        <button
          className="flex items-center justify-center"
          style={{
            width: 22,
            height: 22,
            color: "var(--plugin-label-color)",
          }}
          title="Remove color label"
        >
          <CircleOff size={13} />
        </button>
      </div>

      {/* Center: thumbs per row */}
      <div
        className="flex items-center gap-1"
        style={{
          backgroundColor: "var(--bg-color)",
          borderRadius: 10,
          padding: "2px 8px",
        }}
      >
        <span
          className="text-xs"
          style={{
            color: "var(--fg-color)",
            minWidth: 18,
            textAlign: "center",
          }}
        >
          {gridColumns}
        </span>
        <button
          onClick={fewer}
          className="flex items-center justify-center"
          style={{
            width: 22,
            height: 22,
            color: "var(--plugin-label-color)",
          }}
          title="Fewer thumbnails per row"
        >
          <ZoomOut size={14} />
        </button>
        <button
          onClick={more}
          className="flex items-center justify-center"
          style={{
            width: 22,
            height: 22,
            color: "var(--plugin-label-color)",
          }}
          title="More thumbnails per row"
        >
          <ZoomIn size={14} />
        </button>
      </div>
    </div>
  );
}
