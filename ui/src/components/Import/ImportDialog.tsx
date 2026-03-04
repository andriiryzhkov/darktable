import { useCallback, useRef } from "react";
import { useImportStore } from "../../stores/importStore";
import DialogOverlay from "../DialogOverlay";
import PlacesList from "./PlacesList";
import FolderTree from "./FolderTree";
import FileList from "./FileList";
import ImportBottomBar from "./ImportBottomBar";
import ModuleCheckbox from "../Sidebar/controls/ModuleCheckbox";

export default function ImportDialog() {
  const isOpen = useImportStore((s) => s.isOpen);
  const importMode = useImportStore((s) => s.importMode);
  const closeDialog = useImportStore((s) => s.closeDialog);
  const leftPanelWidth = useImportStore((s) => s.leftPanelWidth);
  const setLeftPanelWidth = useImportStore((s) => s.setLeftPanelWidth);

  const selectOnlyNew = useImportStore((s) => s.selectOnlyNew);
  const recursive = useImportStore((s) => s.recursive);
  const ignoreNonRaw = useImportStore((s) => s.ignoreNonRaw);
  const setSelectOnlyNew = useImportStore((s) => s.setSelectOnlyNew);
  const setRecursive = useImportStore((s) => s.setRecursive);
  const setIgnoreNonRaw = useImportStore((s) => s.setIgnoreNonRaw);

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

  if (!isOpen) return null;

  return (
    <DialogOverlay
      title={importMode === "copy" ? "copy & import" : "add to library"}
      onClose={closeDialog}
    >
      <div className="import-dialog">
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
    </DialogOverlay>
  );
}
