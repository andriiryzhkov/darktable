import { useCallback } from "react";
import { useImportStore } from "../../stores/importStore";
import { pickFolder } from "../../api/commands";
import { Plus, Minus, RotateCcw } from "lucide-react";

export default function PlacesList() {
  const places = useImportStore((s) => s.places);
  const selectedPlacePath = useImportStore((s) => s.selectedPlacePath);
  const selectPlace = useImportStore((s) => s.selectPlace);
  const addCustomPlace = useImportStore((s) => s.addCustomPlace);
  const removePlace = useImportStore((s) => s.removePlace);
  const openDialog = useImportStore((s) => s.openDialog);

  const handleAddPlace = useCallback(async () => {
    try {
      const path = await pickFolder();
      if (path) addCustomPlace(path);
    } catch {
      // pickFolder not available (mock mode) — ignore
    }
  }, [addCustomPlace]);

  const handleRemovePlace = useCallback(() => {
    if (selectedPlacePath) removePlace(selectedPlacePath);
  }, [selectedPlacePath, removePlace]);

  const handleReset = useCallback(() => {
    // Re-open resets to default places
    openDialog();
  }, [openDialog]);

  return (
    <div className="import-places">
      {/* Header */}
      <div className="import-section-header">
        <span className="module-section-title">places</span>
        <div className="import-section-actions">
          <button className="module-action-btn" title="add place" onClick={handleAddPlace}>
            <Plus size={12} />
          </button>
          <button className="module-action-btn" title="remove place" onClick={handleRemovePlace}>
            <Minus size={12} />
          </button>
          <button className="module-action-btn" title="reset places" onClick={handleReset}>
            <RotateCcw size={11} />
          </button>
        </div>
      </div>

      {/* List */}
      <div className="import-section-list">
        {places.map((place) => (
          <div
            key={place.path}
            className="import-place-item"
            data-selected={place.path === selectedPlacePath || undefined}
            onClick={() => selectPlace(place.path)}
            title={place.path}
          >
            {place.name}
          </div>
        ))}
      </div>
    </div>
  );
}
