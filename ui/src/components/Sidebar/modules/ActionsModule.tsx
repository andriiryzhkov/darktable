import CollapsibleModule from "../CollapsibleModule";
import {
  Trash2,
  Move,
  Copy,
  RotateCw,
  RotateCcw,
  Group,
  Ungroup,
  FolderOpen,
} from "lucide-react";

function ActionButton({
  label,
  icon: Icon,
}: {
  label: string;
  icon?: React.ElementType;
}) {
  return (
    <button
      className="flex items-center justify-center gap-1 px-2 py-1 text-xs rounded"
      style={{
        backgroundColor: "var(--button-bg)",
        color: "var(--button-fg)",
        border: "1px solid var(--button-border)",
      }}
      title={label}
    >
      {Icon && <Icon size={12} />}
      <span>{label}</span>
    </button>
  );
}

export default function ActionsModule() {
  return (
    <CollapsibleModule title="actions on selection" defaultOpen>
      <div className="space-y-2">
        {/* Tab bar */}
        <div className="flex gap-1 text-xs">
          <button
            className="flex-1 py-1 rounded"
            style={{
              backgroundColor: "var(--collapsible-bg-color)",
              color: "var(--fg-color)",
            }}
          >
            images
          </button>
          <button
            className="flex-1 py-1 rounded"
            style={{
              backgroundColor: "var(--plugin-bg-color)",
              color: "var(--plugin-label-color)",
            }}
          >
            metadata
          </button>
        </div>

        {/* Action buttons grid */}
        <div className="grid grid-cols-2 gap-1">
          <ActionButton label="Remove" />
          <ActionButton label="delete (trash)" icon={Trash2} />
          <ActionButton label="move..." icon={Move} />
          <ActionButton label="copy..." icon={Copy} />
          <ActionButton label="create HDR" />
          <ActionButton label="duplicate" />
        </div>

        {/* Rotation */}
        <div className="flex gap-1">
          <button
            className="flex-1 flex items-center justify-center py-1 rounded"
            style={{
              backgroundColor: "var(--button-bg)",
              color: "var(--button-fg)",
              border: "1px solid var(--button-border)",
            }}
            title="Rotate counter-clockwise"
          >
            <RotateCcw size={12} />
          </button>
          <button
            className="flex-1 flex items-center justify-center py-1 rounded"
            style={{
              backgroundColor: "var(--button-bg)",
              color: "var(--button-fg)",
              border: "1px solid var(--button-border)",
            }}
            title="Rotate clockwise"
          >
            <RotateCw size={12} />
          </button>
        </div>
        <div className="text-xs text-center" style={{ color: "var(--plugin-label-color)" }}>
          reset rotation
        </div>

        {/* More actions */}
        <div className="grid grid-cols-2 gap-1">
          <ActionButton label="copy locally" />
          <ActionButton label="resync local copy" />
          <ActionButton label="group" icon={Group} />
          <ActionButton label="ungroup" icon={Ungroup} />
        </div>

        <button
          className="w-full flex items-center justify-center gap-1 py-1 text-xs rounded"
          style={{
            backgroundColor: "var(--button-bg)",
            color: "var(--button-fg)",
            border: "1px solid var(--button-border)",
          }}
        >
          <FolderOpen size={12} />
          <span>show in files</span>
        </button>
      </div>
    </CollapsibleModule>
  );
}
