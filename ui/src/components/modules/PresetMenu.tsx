import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { createPortal } from "react-dom";
import { Check, Trash2, ChevronRight, ChevronLeft } from "lucide-react";
import { useDevelopStore } from "../../stores/developStore";
import { useCatalogStore } from "../../stores/catalogStore";
import type { PresetInfo } from "../../types/protocol";
import StorePresetDialog from "./StorePresetDialog";
import type { PresetFilterParams, ImageDefaults } from "./StorePresetDialog";

interface TreeNode {
  label: string;
  fullPath: string;
  preset?: PresetInfo;
  children: TreeNode[];
}

/** Strip "_builtin_" prefix from segment labels */
function cleanLabel(s: string): string {
  return s.startsWith("_builtin_") ? s.slice(9) : s;
}

function buildTree(presets: PresetInfo[]): TreeNode {
  const root: TreeNode = { label: "", fullPath: "", children: [] };

  for (const p of presets) {
    const parts = p.name.split("|");
    let node = root;

    for (let i = 0; i < parts.length - 1; i++) {
      const key = parts[i];
      let child = node.children.find((c) => !c.preset && c.label === cleanLabel(key));
      if (!child) {
        child = { label: cleanLabel(key), fullPath: parts.slice(0, i + 1).join("|"), children: [] };
        node.children.push(child);
      }
      node = child;
    }

    node.children.push({
      label: cleanLabel(parts[parts.length - 1]),
      fullPath: p.name,
      preset: p,
      children: [],
    });
  }

  return root;
}

interface Props {
  op: string;
  moduleName: string;
  anchorRef: React.RefObject<HTMLElement | null>;
  onClose: () => void;
}

export default function PresetMenu({ op, moduleName, anchorRef, onClose }: Props) {
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [showStoreDialog, setShowStoreDialog] = useState(false);
  const [path, setPath] = useState<TreeNode[]>([]);
  const listPresets = useDevelopStore((s) => s.listPresets);
  const applyPreset = useDevelopStore((s) => s.applyPreset);
  const storePreset = useDevelopStore((s) => s.storePreset);
  const removePreset = useDevelopStore((s) => s.removePreset);
  const imgid = useDevelopStore((s) => s.imgid);
  const images = useCatalogStore((s) => s.images);
  const popupRef = useRef<HTMLDivElement>(null);

  const imageDefaults = useMemo<ImageDefaults | undefined>(() => {
    if (!imgid) return undefined;
    const img = images.find((i) => i.id === imgid);
    if (!img) return undefined;
    return {
      maker: img.maker,
      model: img.model,
      lens: img.lens,
      iso: img.iso,
      exposure: img.exposure,
      aperture: img.aperture,
      focal_length: img.focal_length,
    };
  }, [imgid, images]);

  const refresh = useCallback(async () => {
    const result = await listPresets(op);
    setPresets(result);
    setLoading(false);
  }, [op, listPresets]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const tree = buildTree(presets);
  const currentNode = path.reduce<TreeNode | null>((node, step) => {
    if (!node) return null;
    return node.children.find((c) => c.fullPath === step.fullPath) ?? null;
  }, tree) ?? tree;

  const [pos, setPos] = useState({ top: 0, left: 0 });
  useEffect(() => {
    const el = anchorRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    setPos({ top: rect.bottom + 2, left: rect.right });
  }, [anchorRef]);

  // Close on outside click or Escape (but not when dialog is open)
  useEffect(() => {
    if (showStoreDialog) return;
    const onDown = (e: MouseEvent) => {
      if (popupRef.current && !popupRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (path.length > 0) {
          setPath((p) => p.slice(0, -1));
        } else {
          onClose();
        }
      }
    };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [onClose, path.length, showStoreDialog]);

  const handleApply = async (name: string) => {
    await applyPreset(op, name);
    onClose();
  };

  const handleDelete = async (e: React.MouseEvent, name: string) => {
    e.stopPropagation();
    await removePreset(op, name);
    refresh();
  };

  const handleStoreConfirm = async (name: string, description: string, filters?: PresetFilterParams) => {
    await storePreset(op, name, description, filters);
    setShowStoreDialog(false);
    onClose();
  };

  const drillInto = (node: TreeNode) => {
    setPath((p) => [...p, node]);
  };

  const goBack = () => {
    setPath((p) => p.slice(0, -1));
  };

  return (
    <>
      {createPortal(
        <div
          ref={popupRef}
          className="bauhaus-combo-popup preset-menu"
          style={{ top: pos.top, left: pos.left, transform: "translateX(-100%)", minWidth: 180, maxHeight: 400, overflowY: "auto" }}
        >
          {loading ? (
            <div className="preset-menu-empty">loading...</div>
          ) : (
            <>
              {path.length > 0 && (
                <div className="preset-menu-back" onClick={goBack}>
                  <ChevronLeft size={10} />
                  <span>{currentNode.label || "back"}</span>
                </div>
              )}

              {currentNode.children.length === 0 && presets.length === 0 && (
                <div className="preset-menu-empty">no presets</div>
              )}

              {currentNode.children.map((node) => {
                if (node.preset) {
                  const p = node.preset;
                  return (
                    <div
                      key={node.fullPath}
                      className="preset-menu-item"
                      data-active={p.active}
                      onClick={() => handleApply(p.name)}
                    >
                      <span className="preset-menu-check">{p.active ? <Check size={10} /> : null}</span>
                      <span className="preset-menu-label">{node.label}</span>
                      {!p.writeprotect && (
                        <span className="preset-menu-delete" onClick={(e) => handleDelete(e, p.name)}>
                          <Trash2 size={10} />
                        </span>
                      )}
                    </div>
                  );
                }

                return (
                  <div
                    key={node.fullPath}
                    className="preset-menu-item preset-menu-folder"
                    onClick={() => drillInto(node)}
                  >
                    <span className="preset-menu-check" />
                    <span className="preset-menu-label">{node.label}</span>
                    <ChevronRight size={10} className="preset-menu-chevron" />
                  </div>
                );
              })}

              {path.length === 0 && (
                <>
                  <div className="preset-menu-separator" />
                  <div className="preset-menu-item" onClick={() => setShowStoreDialog(true)}>
                    <span className="preset-menu-check" />
                    <span className="preset-menu-label">store new preset...</span>
                  </div>
                </>
              )}
            </>
          )}
        </div>,
        document.body,
      )}
      {showStoreDialog && (
        <StorePresetDialog
          moduleName={moduleName}
          imageDefaults={imageDefaults}
          onConfirm={handleStoreConfirm}
          onCancel={() => setShowStoreDialog(false)}
        />
      )}
    </>
  );
}
