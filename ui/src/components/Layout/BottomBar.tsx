import {
  Star,
  Grid3x3,
  List,
  Columns3,
  ZoomIn,
  ZoomOut,
  Minus,
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
  const { thumbnailSize, setThumbnailSize } = useUIStore();
  const total = useCatalogStore((s) => s.total);

  return (
    <div
      className="flex items-center justify-between px-3 shrink-0"
      style={{
        height: 32,
        backgroundColor: "var(--plugin-bg-color)",
        borderTop: "1px solid var(--border-color)",
      }}
    >
      {/* Left: rating + color labels */}
      <div className="flex items-center gap-1">
        {/* Reject */}
        <button
          className="flex items-center justify-center"
          style={{
            width: 20,
            height: 20,
            color: "var(--plugin-label-color)",
          }}
          title="Reject"
        >
          <Minus size={12} />
        </button>

        {/* Stars */}
        {[1, 2, 3, 4, 5].map((n) => (
          <button
            key={n}
            className="flex items-center justify-center"
            style={{
              width: 20,
              height: 20,
              color: "var(--plugin-label-color)",
            }}
            title={`Rate ${n} stars`}
          >
            <Star size={12} />
          </button>
        ))}

        {/* Separator */}
        <div
          className="mx-2"
          style={{
            width: 1,
            height: 16,
            backgroundColor: "var(--border-color)",
          }}
        />

        {/* Color labels */}
        {COLORS.map((c) => (
          <button
            key={c.key}
            className="flex items-center justify-center"
            style={{ width: 20, height: 20 }}
            title={`Color label: ${c.key}`}
          >
            <div
              className="rounded-full"
              style={{
                width: 10,
                height: 10,
                backgroundColor: `var(${c.var})`,
              }}
            />
          </button>
        ))}

        {/* Remove color label */}
        <button
          className="flex items-center justify-center"
          style={{
            width: 20,
            height: 20,
            color: "var(--plugin-label-color)",
          }}
          title="Remove color label"
        >
          <Minus size={10} />
        </button>
      </div>

      {/* Center: view mode */}
      <div className="flex items-center gap-1">
        <button
          className="flex items-center justify-center"
          style={{
            width: 24,
            height: 24,
            color: "var(--fg-color)",
          }}
          title="Grid view"
        >
          <Grid3x3 size={14} />
        </button>
        <button
          className="flex items-center justify-center"
          style={{
            width: 24,
            height: 24,
            color: "var(--plugin-label-color)",
          }}
          title="List view"
        >
          <List size={14} />
        </button>
        <button
          className="flex items-center justify-center"
          style={{
            width: 24,
            height: 24,
            color: "var(--plugin-label-color)",
          }}
          title="Filmstrip view"
        >
          <Columns3 size={14} />
        </button>
      </div>

      {/* Right: count + zoom */}
      <div className="flex items-center gap-2">
        <span
          className="text-xs"
          style={{ color: "var(--plugin-label-color)" }}
        >
          {total}
        </span>

        <button
          onClick={() => setThumbnailSize(thumbnailSize - 20)}
          className="flex items-center justify-center"
          style={{
            width: 20,
            height: 20,
            color: "var(--plugin-label-color)",
          }}
          title="Zoom out"
        >
          <ZoomOut size={12} />
        </button>

        <input
          type="range"
          min={100}
          max={400}
          value={thumbnailSize}
          onChange={(e) => setThumbnailSize(Number(e.target.value))}
          className="dt-slider"
          style={{ width: 80 }}
        />

        <button
          onClick={() => setThumbnailSize(thumbnailSize + 20)}
          className="flex items-center justify-center"
          style={{
            width: 20,
            height: 20,
            color: "var(--plugin-label-color)",
          }}
          title="Zoom in"
        >
          <ZoomIn size={12} />
        </button>
      </div>
    </div>
  );
}
