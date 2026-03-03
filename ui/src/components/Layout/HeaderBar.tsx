import { useUIStore } from "../../stores/uiStore";
import logoSrc from "../../assets/darktable-logo.png";

export default function HeaderBar() {
  const { activeView, setActiveView } = useUIStore();

  return (
    <div className="headerbar">
      {/* Left: logo + title + version */}
      <div className="flex items-center gap-2">
        <img className="headerbar-logo" src={logoSrc} alt="darktable" />
        <span className="headerbar-title">darktable</span>
        <span className="headerbar-version">5.5.0</span>
      </div>

      {/* Right: view tabs */}
      <div className="headerbar-tabs">
        {(["lighttable", "darkroom"] as const).map((view) => (
          <button
            key={view}
            onClick={() => setActiveView(view)}
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
