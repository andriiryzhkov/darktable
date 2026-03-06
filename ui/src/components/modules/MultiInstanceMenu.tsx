import { useState, useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { useDevelopStore } from "../../stores/developStore";
import { IOP_FLAGS } from "../../types/protocol";

interface Props {
  op: string;
  instance: number;
  anchorRef: React.RefObject<HTMLElement | null>;
  onClose: () => void;
  onRename: () => void;
}

export default function MultiInstanceMenu({ op, instance, anchorRef, onClose, onRename }: Props) {
  const modules = useDevelopStore((s) => s.modules);
  const newInstance = useDevelopStore((s) => s.newInstance);
  const deleteInstance = useDevelopStore((s) => s.deleteInstance);
  const moveInstance = useDevelopStore((s) => s.moveInstance);
  const popupRef = useRef<HTMLDivElement>(null);

  // Compute multi_show state from modules list
  const moduleInfo = modules.find((m) => m.op === op && m.instance === instance);
  const sameOpModules = modules.filter((m) => m.op === op);
  const nbInstances = sameOpModules.length;
  const canNew = moduleInfo ? !(moduleInfo.flags & IOP_FLAGS.ONE_INSTANCE) : false;
  const canDelete = nbInstances > 1;

  // Determine move availability based on position in iop list
  const moduleIndex = modules.findIndex((m) => m.op === op && m.instance === instance);
  const canMoveUp = moduleIndex < modules.length - 1;
  const canMoveDown = moduleIndex > 0;

  const [pos, setPos] = useState({ top: 0, left: 0 });
  useEffect(() => {
    const el = anchorRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    setPos({ top: rect.bottom + 2, left: rect.right });
  }, [anchorRef]);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (popupRef.current && !popupRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  const handleNew = async () => {
    await newInstance(op, instance, false);
    onClose();
  };

  const handleDuplicate = async () => {
    await newInstance(op, instance, true);
    onClose();
  };

  const handleMoveUp = async () => {
    await moveInstance(op, instance, "up");
    onClose();
  };

  const handleMoveDown = async () => {
    await moveInstance(op, instance, "down");
    onClose();
  };

  const handleDelete = async () => {
    await deleteInstance(op, instance);
    onClose();
  };

  return createPortal(
    <div
      ref={popupRef}
      className="bauhaus-combo-popup multi-instance-menu"
      style={{ top: pos.top, left: pos.left, transform: "translateX(-100%)", minWidth: 160 }}
    >
      <div className={`preset-menu-item${canNew ? "" : " disabled"}`} onClick={canNew ? handleNew : undefined}>
        <span className="preset-menu-check" />
        <span className="preset-menu-label">new instance</span>
      </div>
      <div className={`preset-menu-item${canNew ? "" : " disabled"}`} onClick={canNew ? handleDuplicate : undefined}>
        <span className="preset-menu-check" />
        <span className="preset-menu-label">duplicate instance</span>
      </div>
      <div className="preset-menu-separator" />
      <div className={`preset-menu-item${canMoveUp ? "" : " disabled"}`} onClick={canMoveUp ? handleMoveUp : undefined}>
        <span className="preset-menu-check" />
        <span className="preset-menu-label">move up</span>
      </div>
      <div className={`preset-menu-item${canMoveDown ? "" : " disabled"}`} onClick={canMoveDown ? handleMoveDown : undefined}>
        <span className="preset-menu-check" />
        <span className="preset-menu-label">move down</span>
      </div>
      <div className="preset-menu-separator" />
      <div className="preset-menu-item" onClick={onRename}>
        <span className="preset-menu-check" />
        <span className="preset-menu-label">rename</span>
      </div>
      <div className={`preset-menu-item${canDelete ? "" : " disabled"}`} onClick={canDelete ? handleDelete : undefined}>
        <span className="preset-menu-check" />
        <span className="preset-menu-label">delete</span>
      </div>
    </div>,
    document.body,
  );
}
