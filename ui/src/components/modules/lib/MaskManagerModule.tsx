import { useState, useCallback, useEffect, useRef, useMemo } from "react";
import { createPortal } from "react-dom";
import { Circle, SplinePointer, ArrowDownRight, Brush, Layers, ChevronRight, ChevronDown } from "lucide-react";
import LibModuleCard from "../LibModuleCard";
import BauhausButton from "../../controls/BauhausButton";
import BauhausTooltip from "../../controls/BauhausTooltip";
import BauhausSlider from "../../controls/BauhausSlider";
import BauhausCollapsible from "../../controls/BauhausCollapsible";
import { useDevelopStore } from "../../../stores/developStore";
import { MASKS_TYPE } from "../../../types/protocol";
import type { MaskForm, MaskUsage } from "../../../types/protocol";

const ICON_SIZE = 12;

function MaskTypeIcon({ type }: { type: number }) {
  const base = type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
  switch (base) {
    case MASKS_TYPE.CIRCLE:
      return <Circle size={ICON_SIZE} />;
    case MASKS_TYPE.ELLIPSE:
      return <Circle size={ICON_SIZE} style={{ transform: "scaleX(0.7)" }} />;
    case MASKS_TYPE.PATH:
      return <SplinePointer size={ICON_SIZE} />;
    case MASKS_TYPE.GRADIENT:
      return <ArrowDownRight size={ICON_SIZE} />;
    case MASKS_TYPE.BRUSH:
      return <Brush size={ICON_SIZE} />;
    case MASKS_TYPE.GROUP:
      return <Layers size={ICON_SIZE} />;
    default:
      return <Circle size={ICON_SIZE} />;
  }
}

function MaskItem({
  form,
  indent,
  selected,
  onSelect,
  onRename,
  onContextMenu,
}: {
  form: MaskForm;
  indent?: boolean;
  selected?: boolean;
  onSelect?: (formid: number | null) => void;
  onRename: (formid: number, name: string) => void;
  onContextMenu: (formid: number, e: React.MouseEvent) => void;
}) {
  const [renaming, setRenaming] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleStartRename = useCallback(() => {
    setRenameValue(form.name);
    setRenaming(true);
  }, [form.name]);

  const handleSubmit = useCallback(() => {
    setRenaming(false);
    const trimmed = renameValue.trim();
    if (trimmed && trimmed !== form.name) {
      onRename(form.formid, trimmed);
    }
  }, [form.formid, form.name, renameValue, onRename]);

  useEffect(() => {
    if (renaming && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [renaming]);

  return (
    <div
      className="mask-item"
      data-indent={indent || undefined}
      data-selected={selected || undefined}
      onClick={(e) => { e.stopPropagation(); onSelect?.(selected ? null : form.formid); }}
      onContextMenu={(e) => { e.preventDefault(); onContextMenu(form.formid, e); }}
    >
      <span className="mask-item-icon">
        <MaskTypeIcon type={form.type} />
      </span>
      {renaming ? (
        <input
          ref={inputRef}
          className="mask-item-rename"
          value={renameValue}
          onChange={(e) => setRenameValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") { e.preventDefault(); handleSubmit(); }
            else if (e.key === "Escape") { e.preventDefault(); setRenaming(false); }
          }}
          onBlur={handleSubmit}
        />
      ) : (
        <span className="mask-item-name" onDoubleClick={handleStartRename}>
          {form.name}
          {form.is_clone && <span className="mask-clone-badge">clone</span>}
        </span>
      )}
    </div>
  );
}

/** A module group header with expandable children */
function ModuleGroup({
  usage,
  forms,
  formsById,
  selectedMaskId,
  onSelect,
  onRename,
  onContextMenu,
}: {
  usage: MaskUsage;
  forms: MaskForm | undefined;
  formsById: Map<number, MaskForm>;
  selectedMaskId: number | null;
  onSelect: (formid: number | null) => void;
  onRename: (formid: number, name: string) => void;
  onContextMenu: (formid: number, e: React.MouseEvent) => void;
}) {
  const [expanded, setExpanded] = useState(true);
  const children = forms?.children ?? [];
  const label = usage.module_name ?? usage.op;

  return (
    <div className="mask-group">
      <div className="mask-group-header" onClick={() => setExpanded(!expanded)}>
        {expanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
        <Layers size={ICON_SIZE} />
        <span className="mask-group-name">{label}</span>
        <span className="mask-group-count">{children.length}</span>
      </div>
      {expanded && children.length > 0 && (
        <div className="mask-group-children">
          {children.map((child) => {
            const childForm = formsById.get(child.formid);
            if (!childForm) return null;
            return (
              <MaskItem
                key={child.formid}
                form={childForm}
                indent
                selected={selectedMaskId === child.formid}
                onSelect={onSelect}
                onRename={onRename}
                onContextMenu={onContextMenu}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function MaskManagerModule() {
  const sessionId = useDevelopStore((s) => s.sessionId);
  const maskForms = useDevelopStore((s) => s.maskForms);
  const maskUsage = useDevelopStore((s) => s.maskUsage);
  const selectedMaskId = useDevelopStore((s) => s.selectedMaskId);
  const fetchMasks = useDevelopStore((s) => s.fetchMasks);
  const renameMask = useDevelopStore((s) => s.renameMask);
  const deleteMask = useDevelopStore((s) => s.deleteMask);
  const selectMask = useDevelopStore((s) => s.selectMask);

  // Context menu state
  const [menuPos, setMenuPos] = useState<{ top: number; left: number } | null>(null);
  const [menuFormId, setMenuFormId] = useState<number | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (sessionId) fetchMasks();
  }, [sessionId, fetchMasks]);

  // Close context menu on outside click
  useEffect(() => {
    if (!menuPos) return;
    const close = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuPos(null);
      }
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [menuPos]);

  const handleMaskContextMenu = useCallback((formid: number, e: React.MouseEvent) => {
    setMenuFormId(formid);
    setMenuPos({ top: e.clientY, left: e.clientX });
  }, []);

  const handleTreeContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setMenuFormId(null);
    setMenuPos({ top: e.clientY, left: e.clientX });
  }, []);

  const handleDeleteShape = useCallback(() => {
    if (menuFormId !== null) deleteMask(menuFormId);
    setMenuPos(null);
  }, [menuFormId, deleteMask]);

  // Build form lookup by id
  const formsById = useMemo(() => {
    const map = new Map<number, MaskForm>();
    for (const f of maskForms) map.set(f.formid, f);
    return map;
  }, [maskForms]);

  // Forms used by modules (group forms referenced by blend_params)
  const moduleGroupIds = useMemo(
    () => new Set(maskUsage.map((u) => u.mask_id)),
    [maskUsage],
  );

  // Forms that are children of module groups
  const childIds = useMemo(() => {
    const ids = new Set<number>();
    for (const u of maskUsage) {
      const grp = formsById.get(u.mask_id);
      if (grp?.children) {
        for (const c of grp.children) ids.add(c.formid);
      }
    }
    return ids;
  }, [maskUsage, formsById]);

  // Unused shapes: not a module group and not a child of any module group
  const unusedForms = useMemo(
    () => maskForms.filter((f) => !moduleGroupIds.has(f.formid) && !childIds.has(f.formid)),
    [maskForms, moduleGroupIds, childIds],
  );

  const handleDeleteUnused = useCallback(() => {
    for (const f of unusedForms) deleteMask(f.formid);
    setMenuPos(null);
  }, [unusedForms, deleteMask]);

  if (!sessionId) {
    return (
      <LibModuleCard
        title="mask manager"
        description="manage drawn masks used by processing modules"
      >
        <p className="mask-empty">no active session</p>
      </LibModuleCard>
    );
  }

  return (
    <>
      <LibModuleCard
        title="mask manager"
        description="manage drawn masks used by processing modules"
      >
        {/* Shape creation toolbar */}
        <div className="mask-toolbar">
          <span className="mask-toolbar-label">created shapes</span>
          <span className="mask-toolbar-buttons">
            <BauhausTooltip content="add brush" placement="bottom">
              <BauhausButton icon={<Brush size={ICON_SIZE} />} transparent disabled />
            </BauhausTooltip>
            <BauhausTooltip content="add circle" placement="bottom">
              <BauhausButton icon={<Circle size={ICON_SIZE} />} transparent disabled />
            </BauhausTooltip>
            <BauhausTooltip content="add ellipse" placement="bottom">
              <BauhausButton icon={<Circle size={ICON_SIZE} style={{ transform: "scaleX(0.7)" }} />} transparent disabled />
            </BauhausTooltip>
            <BauhausTooltip content="add path" placement="bottom">
              <BauhausButton icon={<SplinePointer size={ICON_SIZE} />} transparent disabled />
            </BauhausTooltip>
            <BauhausTooltip content="add gradient" placement="bottom">
              <BauhausButton icon={<ArrowDownRight size={ICON_SIZE} />} transparent disabled />
            </BauhausTooltip>
          </span>
        </div>

        {/* Mask tree: grouped by module, then unused */}
        <div className="mask-tree" onContextMenu={handleTreeContextMenu} onClick={() => selectMask(null)}>
          {maskUsage.map((u) => (
            <ModuleGroup
              key={`${u.op}-${u.instance}`}
              usage={u}
              forms={formsById.get(u.mask_id)}
              formsById={formsById}
              selectedMaskId={selectedMaskId}
              onSelect={selectMask}
              onRename={renameMask}
              onContextMenu={handleMaskContextMenu}
            />
          ))}

          {unusedForms.length > 0 && (
            <>
              {maskUsage.length > 0 && <div className="mask-separator" />}
              {unusedForms.map((form) => (
                <MaskItem
                  key={form.formid}
                  form={form}
                  selected={selectedMaskId === form.formid}
                  onSelect={selectMask}
                  onRename={renameMask}
                  onContextMenu={handleMaskContextMenu}
                />
              ))}
            </>
          )}
        </div>

        {/* Properties section */}
        <BauhausCollapsible title="properties">
          <BauhausSlider
            label="opacity"
            value={1}
            min={0}
            max={1}
            step={0.01}
            defaultValue={1}
            format={(v) => `${(v * 100).toFixed(2)}%`}
          />
          <BauhausSlider
            label="size"
            value={0.05}
            min={0.0001}
            max={1}
            step={0.001}
            defaultValue={0.05}
            format={(v) => `${(v * 100).toFixed(2)}%`}
          />
          <BauhausSlider
            label="feather"
            value={0.33}
            min={0.0001}
            max={1}
            step={0.001}
            defaultValue={0.33}
            format={(v) => `${(v * 100).toFixed(2)}%`}
          />
        </BauhausCollapsible>
      </LibModuleCard>
      {menuPos && createPortal(
        <div
          ref={menuRef}
          className="mask-context-menu"
          style={{ top: menuPos.top, left: menuPos.left }}
        >
          {menuFormId !== null ? (
            <>
              <div className="mask-context-menu-item" onClick={() => setMenuPos(null)}>
                duplicate this shape
              </div>
              <div className="mask-context-menu-item" onClick={handleDeleteShape}>
                delete this shape
              </div>
              <div className="mask-context-menu-divider" />
              <div className="mask-context-menu-item" onClick={handleDeleteUnused}>
                delete unused shapes
              </div>
            </>
          ) : (
            <>
              <div className="mask-context-menu-item" onClick={() => setMenuPos(null)}>
                add brush
              </div>
              <div className="mask-context-menu-item" onClick={() => setMenuPos(null)}>
                add circle
              </div>
              <div className="mask-context-menu-item" onClick={() => setMenuPos(null)}>
                add ellipse
              </div>
              <div className="mask-context-menu-item" onClick={() => setMenuPos(null)}>
                add path
              </div>
              <div className="mask-context-menu-item" onClick={() => setMenuPos(null)}>
                add gradient
              </div>
              <div className="mask-context-menu-divider" />
              <div className="mask-context-menu-item" onClick={handleDeleteUnused}>
                delete unused shapes
              </div>
            </>
          )}
        </div>,
        document.body,
      )}
    </>
  );
}
