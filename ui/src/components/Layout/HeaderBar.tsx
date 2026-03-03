import { useUIStore } from "../../stores/uiStore";
import logoSrc from "../../assets/darktable-logo.png";

export default function HeaderBar() {
  const { activeView, setActiveView } = useUIStore();

  return (
    <div
      className="flex items-center justify-between px-4 shrink-0"
      style={{
        height: 40,
        backgroundColor: "var(--plugin-bg-color)",
        borderBottom: "1px solid var(--border-color)",
      }}
    >
      {/* Left: logo + title + version */}
      <div className="flex items-center gap-2">
        <img src={logoSrc} alt="darktable" width={22} height={22} />
        <span
          className="text-sm"
          style={{ color: "var(--fg-color)", fontWeight: 700 }}
        >
          darktable
        </span>
        <span
          style={{
            color: "var(--plugin-label-color)",
            fontSize: "0.7em",
          }}
        >
          5.5.0
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
