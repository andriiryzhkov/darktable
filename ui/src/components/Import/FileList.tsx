import { useState, useCallback } from "react";
import { useImportStore } from "../../stores/importStore";
import { getFileThumbnail } from "../../api/commands";
import { Eye } from "lucide-react";
import type { SortField } from "../../types/import";

function formatDateTime(epoch: number): string {
  const d = new Date(epoch * 1000);
  const date = d.toLocaleDateString("en-CA"); // YYYY-MM-DD
  const time = d.toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  return `${date} ${time}`;
}

function SortIndicator({ field, currentField, dir }: {
  field: SortField;
  currentField: SortField;
  dir: string;
}) {
  if (field !== currentField) return null;
  return (
    <span className="import-sort-indicator">
      {dir === "asc" ? "\u25B3" : "\u25BD"}
    </span>
  );
}

function ThumbCell({ fullpath }: { fullpath: string }) {
  const [state, setState] = useState<"idle" | "loading" | "loaded" | "error">("idle");
  const [dataUrl, setDataUrl] = useState<string | null>(null);

  const handleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation(); // don't toggle file selection
    if (state === "loaded") {
      // Toggle off
      setState("idle");
      setDataUrl(null);
      return;
    }
    if (state === "loading") return;
    setState("loading");
    getFileThumbnail(fullpath)
      .then((result) => {
        setDataUrl(`data:${result.mime};base64,${result.data}`);
        setState("loaded");
      })
      .catch(() => {
        setState("error");
      });
  }, [fullpath, state]);

  return (
    <div className="import-file-col import-file-col-thumb" onClick={handleClick}>
      {state === "idle" && <Eye size={12} />}
      {state === "loading" && <span className="import-thumb-spinner" />}
      {state === "error" && <Eye size={12} />}
      {state === "loaded" && dataUrl && (
        <img className="import-thumb-preview" src={dataUrl} alt="" />
      )}
    </div>
  );
}

export default function FileList() {
  const files = useImportStore((s) => s.files);
  const sortField = useImportStore((s) => s.sortField);
  const sortDir = useImportStore((s) => s.sortDir);
  const setSort = useImportStore((s) => s.setSort);
  const toggleFileSelection = useImportStore((s) => s.toggleFileSelection);
  const selectedFolderPath = useImportStore((s) => s.selectedFolderPath);

  return (
    <div className="import-file-list">
      {/* Header */}
      <div className="import-file-header">
        <div className="import-file-col import-file-col-check">
          <span className="import-file-check-symbol">{"\u2714"}</span>
        </div>
        <div
          className="import-file-col import-file-col-name"
          onClick={() => setSort("name")}
        >
          <span>name</span>
          <SortIndicator field="name" currentField={sortField} dir={sortDir} />
        </div>
        <div
          className="import-file-col import-file-col-modified"
          onClick={() => setSort("modified")}
        >
          <span>modified</span>
          <SortIndicator field="modified" currentField={sortField} dir={sortDir} />
        </div>
        <div className="import-file-col import-file-col-thumb">
          <Eye size={14} />
        </div>
      </div>

      {/* Rows */}
      <div className="import-file-rows">
        {!selectedFolderPath && (
          <div className="import-file-empty">select a folder</div>
        )}
        {files.map((f) => (
          <div
            key={f.fullpath}
            className="import-file-row"
            data-selected={f.selected || undefined}
            onClick={(e) => toggleFileSelection(f.fullpath, e)}
          >
            <div className="import-file-col import-file-col-check">
              {f.alreadyImported && (
                <span className="import-file-check-symbol">{"\u2714"}</span>
              )}
            </div>
            <div className="import-file-col import-file-col-name">
              {f.filename}
            </div>
            <div className="import-file-col import-file-col-modified">
              {formatDateTime(f.modified)}
            </div>
            <ThumbCell fullpath={f.fullpath} />
          </div>
        ))}
      </div>
    </div>
  );
}
