import CollapsibleModule from "../../Sidebar/CollapsibleModule";
import { useDevelopStore } from "../../../stores/developStore";
import ModuleButton from "../../Sidebar/controls/ModuleButton";
import { Archive, Trash2 } from "lucide-react";

export default function DarkroomHistoryModule() {
  const sessionId = useDevelopStore((s) => s.sessionId);

  return (
    <CollapsibleModule title="history stack" defaultOpen>
      {sessionId ? (
        <div>
          <div className="bauhaus-button-row">
            <ModuleButton label="compress" icon={<Archive size={12} />} />
            <ModuleButton label="discard" icon={<Trash2 size={12} />} />
          </div>
          <p
            className="text-xs"
            style={{ color: "var(--fg-color)", padding: "2px 0" }}
          >
            original
          </p>
        </div>
      ) : (
        <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
          no active session
        </p>
      )}
    </CollapsibleModule>
  );
}
