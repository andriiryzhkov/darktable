import { useCallback, useRef, useEffect } from "react";
import { useCatalogStore } from "../../stores/catalogStore";
import { useUIStore } from "../../stores/uiStore";
import ThumbnailCard from "./ThumbnailCard";
import TopToolbar from "./TopToolbar";

interface Props {
  onOpenImage: (imgid: number) => void;
}

export default function LighttableView({ onOpenImage }: Props) {
  const { images, loading, selectedIds, selectImage } = useCatalogStore();
  const thumbnailSize = useUIStore((s) => s.thumbnailSize);
  const setGridColumns = useUIStore((s) => s.setGridColumns);
  const gridRef = useRef<HTMLDivElement>(null);

  const handleSelect = useCallback(
    (id: number, e: React.MouseEvent) => selectImage(id, e),
    [selectImage],
  );

  useEffect(() => {
    const el = gridRef.current;
    if (!el) return;
    const measure = () => {
      const style = window.getComputedStyle(el);
      const cols = style.gridTemplateColumns.split(" ").length;
      setGridColumns(cols);
    };
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, [thumbnailSize, setGridColumns]);

  return (
    <div className="flex flex-col flex-1 min-h-0">
      <TopToolbar />

      {/* Thumbnail grid */}
      <div
        className="flex-1 overflow-y-auto"
        style={{ backgroundColor: "var(--lighttable-bg-color)" }}
      >
        {loading && images.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p style={{ color: "var(--plugin-label-color)" }}>Loading...</p>
          </div>
        ) : (
          <div
            ref={gridRef}
            className="grid gap-0.5"
            style={{
              gridTemplateColumns: `repeat(auto-fill, minmax(${thumbnailSize}px, 1fr))`,
            }}
          >
            {images.map((img) => (
              <ThumbnailCard
                key={img.id}
                imgid={img.id}
                filename={img.filename}
                selected={selectedIds.has(img.id)}
                rating={img.rating ?? 0}
                colorLabels={img.color_labels ?? 0}
                groupId={img.group_id ?? 0}
                localCopy={img.local_copy ?? false}
                altered={img.altered ?? false}
                onSelect={(e) => handleSelect(img.id, e)}
                onDoubleClick={() => onOpenImage(img.id)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
