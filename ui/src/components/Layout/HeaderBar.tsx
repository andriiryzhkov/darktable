import { useUIStore } from "../../stores/uiStore";
import { useCatalogStore } from "../../stores/catalogStore";
import { usePlatform } from "../../hooks/usePlatform";
import { windowStartDrag, windowZoom } from "../../api/commands";
import logoSvg from "../../assets/idbutton.svg";
import titleSvg from "../../assets/darktable.svg";

function isInteractive(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.closest("button, a, input, select") !== null;
}

export default function HeaderBar() {
  const { activeView, setActiveView } = useUIStore();
  const platform = usePlatform();

  return (
    <div
      className={`headerbar${platform === "macos" ? " headerbar--mac" : ""}`}
      onMouseDown={(e) => {
        if (e.button === 0 && !isInteractive(e.target)) windowStartDrag();
      }}
      onDoubleClick={(e) => {
        if (!isInteractive(e.target)) windowZoom();
      }}
    >
      {/* Left: logo + title + version */}
      <div className="flex items-center gap-2">
        <img className="headerbar-logo" src={logoSvg} alt="darktable" />
        <img className="headerbar-title-svg" src={titleSvg} alt="darktable" />
        <span className="headerbar-version">5.5.0</span>
      </div>

      {/* Right: view tabs */}
      <div className="headerbar-tabs">
        {(["lighttable", "darkroom"] as const).map((view) => (
          <button
            key={view}
            onClick={() => {
              if (view === "darkroom" && useCatalogStore.getState().selectedIds.size === 0) return;
              setActiveView(view);
            }}
            className="headerbar-tab"
            data-active={activeView === view}
          >
            {view}
          </button>
        ))}
      </div>
    </div>
  );
}
