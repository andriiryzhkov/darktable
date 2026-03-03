import { useImportStore } from "../../stores/importStore";

export default function ImportBottomBar() {
  const files = useImportStore((s) => s.files);
  const importMode = useImportStore((s) => s.importMode);
  const selectAll = useImportStore((s) => s.selectAll);
  const selectNone = useImportStore((s) => s.selectNone);
  const selectNew = useImportStore((s) => s.selectNew);
  const closeDialog = useImportStore((s) => s.closeDialog);
  const doImport = useImportStore((s) => s.doImport);

  const selectedCount = files.filter((f) => f.selected).length;
  const totalCount = files.length;

  return (
    <div className="import-bottom">
      <div className="import-bottom-left">
        <button className="bauhaus-button" onClick={selectAll}>
          select all
        </button>
        <button className="bauhaus-button" onClick={selectNone}>
          select none
        </button>
        <button className="bauhaus-button" onClick={selectNew}>
          select new
        </button>
      </div>

      <div className="import-bottom-center">
        {totalCount > 0
          ? `${selectedCount} image${selectedCount !== 1 ? "s" : ""} out of ${totalCount} selected`
          : ""}
      </div>

      <div className="import-bottom-right">
        <button className="bauhaus-button" onClick={closeDialog}>
          cancel
        </button>
        <button
          className="bauhaus-button bauhaus-button-primary"
          disabled={selectedCount === 0}
          onClick={doImport}
        >
          {importMode === "copy" ? "copy & import" : "add to library"}
        </button>
      </div>
    </div>
  );
}
