import { useUIStore } from "../../stores/uiStore";

export default function HeaderBar() {
  const { activeView, setActiveView } = useUIStore();

  return (
    <div
      className="flex items-center justify-between px-3 shrink-0"
      style={{
        height: 32,
        backgroundColor: "var(--plugin-bg-color)",
        borderBottom: "1px solid var(--border-color)",
      }}
    >
      {/* Left: logo */}
      <div className="flex items-center gap-2">
        <span
          className="text-sm tracking-wide"
          style={{ color: "var(--fg-color)", fontWeight: 300 }}
        >
          darktable
        </span>
        <span
          className="text-xs"
          style={{ color: "var(--plugin-label-color)" }}
        >
          6.0
        </span>
      </div>

      {/* Right: view tabs */}
      <div className="flex items-center gap-4">
        {(["lighttable", "darkroom"] as const).map((view) => (
          <button
            key={view}
            onClick={() => setActiveView(view)}
            className="text-sm border-none bg-transparent cursor-pointer"
            style={{
              color:
                activeView === view
                  ? "var(--fg-color)"
                  : "var(--plugin-label-color)",
              fontWeight: activeView === view ? 500 : 400,
            }}
          >
            {view}
          </button>
        ))}
      </div>
    </div>
  );
}
