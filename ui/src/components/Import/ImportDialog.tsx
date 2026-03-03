import { useEffect, useCallback, useRef, useState } from "react";
import { useImportStore } from "../../stores/importStore";
import { usePlatform } from "../../hooks/usePlatform";
import WindowControls from "../WindowControls";
import PlacesList from "./PlacesList";
import FolderTree from "./FolderTree";
import FileList from "./FileList";
import ImportBottomBar from "./ImportBottomBar";
import ModuleCheckbox from "../Sidebar/controls/ModuleCheckbox";

export default function ImportDialog() {
  const isOpen = useImportStore((s) => s.isOpen);
  const importMode = useImportStore((s) => s.importMode);
  const closeDialog = useImportStore((s) => s.closeDialog);
  const os = usePlatform();
  const leftPanelWidth = useImportStore((s) => s.leftPanelWidth);
  const setLeftPanelWidth = useImportStore((s) => s.setLeftPanelWidth);

  const selectOnlyNew = useImportStore((s) => s.selectOnlyNew);
  const recursive = useImportStore((s) => s.recursive);
  const ignoreNonRaw = useImportStore((s) => s.ignoreNonRaw);
  const setSelectOnlyNew = useImportStore((s) => s.setSelectOnlyNew);
  const setRecursive = useImportStore((s) => s.setRecursive);
  const setIgnoreNonRaw = useImportStore((s) => s.setIgnoreNonRaw);

  // Dialog position (drag to move)
  const [position, setPosition] = useState<{ x: number; y: number } | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const moveDragging = useRef(false);
  const moveStartX = useRef(0);
  const moveStartY = useRef(0);
  const moveStartPosX = useRef(0);
  const moveStartPosY = useRef(0);

  const onHeaderPointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      moveDragging.current = true;
      moveStartX.current = e.clientX;
      moveStartY.current = e.clientY;
      // If no position yet, compute from current centered layout
      if (dialogRef.current && position === null) {
        const rect = dialogRef.current.getBoundingClientRect();
        moveStartPosX.current = rect.left;
        moveStartPosY.current = rect.top;
      } else {
        moveStartPosX.current = position?.x ?? 0;
        moveStartPosY.current = position?.y ?? 0;
      }
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    },
    [position],
  );

  const onHeaderPointerMove = useCallback((e: React.PointerEvent) => {
    if (!moveDragging.current) return;
    const dx = e.clientX - moveStartX.current;
    const dy = e.clientY - moveStartY.current;
    setPosition({
      x: moveStartPosX.current + dx,
      y: moveStartPosY.current + dy,
    });
  }, []);

  const onHeaderPointerUp = useCallback((e: React.PointerEvent) => {
    if (!moveDragging.current) return;
    moveDragging.current = false;
    const el = e.currentTarget as HTMLElement;
    if (el.hasPointerCapture(e.pointerId)) {
      el.releasePointerCapture(e.pointerId);
    }
  }, []);

  // Reset position when dialog opens
  useEffect(() => {
    if (isOpen) setPosition(null);
  }, [isOpen]);

  // Pane resize state
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const startX = useRef(0);
  const startW = useRef(0);

  const onDividerPointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      dragging.current = true;
      startX.current = e.clientX;
      startW.current = leftPanelWidth;
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    },
    [leftPanelWidth],
  );

  const onDividerPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragging.current) return;
    const dx = e.clientX - startX.current;
    const newW = Math.max(150, Math.min(400, startW.current + dx));
    if (containerRef.current) {
      const left = containerRef.current.querySelector(".import-left") as HTMLElement;
      if (left) left.style.width = `${newW}px`;
    }
  }, []);

  const onDividerPointerUp = useCallback(
    (e: React.PointerEvent) => {
      if (!dragging.current) return;
      dragging.current = false;
      const el = e.currentTarget as HTMLElement;
      if (el.hasPointerCapture(e.pointerId)) {
        el.releasePointerCapture(e.pointerId);
      }
      if (containerRef.current) {
        const left = containerRef.current.querySelector(".import-left") as HTMLElement;
        if (left) setLeftPanelWidth(parseInt(left.style.width, 10));
      }
    },
    [setLeftPanelWidth],
  );

  // Escape to close
  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeDialog();
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [isOpen, closeDialog]);

  if (!isOpen) return null;

  return (
    <div className="import-overlay">
      <div
        ref={dialogRef}
        className="import-dialog"
        style={
          position
            ? { position: "absolute", left: position.x, top: position.y }
            : undefined
        }
      >
        {/* Header — drag to move */}
        <div
          className="import-header"
          onPointerDown={onHeaderPointerDown}
          onPointerMove={onHeaderPointerMove}
          onPointerUp={onHeaderPointerUp}
        >
          {os === "macos" && <WindowControls onClose={closeDialog} />}
          <span className="import-header-title">
            {importMode === "copy" ? "copy & import" : "add to library"}
          </span>
          {os && os !== "macos" && <WindowControls onClose={closeDialog} />}
        </div>

        {/* Body */}
        <div className="import-body" ref={containerRef}>
          {/* Left panel */}
          <div className="import-left" style={{ width: leftPanelWidth }}>
            <PlacesList />
            <FolderTree />
          </div>

          {/* Divider */}
          <div
            className="import-divider"
            onPointerDown={onDividerPointerDown}
            onPointerMove={onDividerPointerMove}
            onPointerUp={onDividerPointerUp}
          />

          {/* Right panel */}
          <div className="import-right">
            {/* Toolbar */}
            <div className="import-toolbar">
              <ModuleCheckbox
                label="select only new images"
                checked={selectOnlyNew}
                align="left"
                onChange={setSelectOnlyNew}
              />
              <ModuleCheckbox
                label="recursive directory"
                checked={recursive}
                align="left"
                onChange={setRecursive}
              />
              <ModuleCheckbox
                label="ignore non-raw images"
                checked={ignoreNonRaw}
                align="left"
                onChange={setIgnoreNonRaw}
              />
            </div>

            {/* File list */}
            <FileList />
          </div>
        </div>

        {/* Bottom bar */}
        <ImportBottomBar />
      </div>
    </div>
  );
}
