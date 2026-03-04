import { useCallback, useRef, useEffect, useMemo } from "react";
import { useCatalogStore } from "../../stores/catalogStore";
import { useFilterStore } from "../../stores/filterStore";
import { useUIStore } from "../../stores/uiStore";
import ThumbnailCard from "./ThumbnailCard";
import TopToolbar from "./TopToolbar";

interface Props {
  onOpenImage: (imgid: number) => void;
}

export default function LighttableView({ onOpenImage }: Props) {
  const { images, loading, selectedIds, selectImage } = useCatalogStore();
  const grouping = useFilterStore((s) => s.grouping);
  const thumbnailSize = useUIStore((s) => s.thumbnailSize);
  const setGridColumns = useUIStore((s) => s.setGridColumns);
  const groupSizes = useMemo(() => {
    const counts = new Map<number, number>();
    for (const img of images) {
      const gid = img.group_id ?? img.id;
      counts.set(gid, (counts.get(gid) ?? 0) + 1);
    }
    return counts;
  }, [images]);
  const displayedImages = useMemo(
    () => grouping ? images.filter((img) => img.id === (img.group_id ?? img.id)) : images,
    [images, grouping],
  );
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
  }, [thumbnailSize, loading, setGridColumns]);

  return (
    <div className="flex flex-col flex-1 min-h-0">
      <TopToolbar />

      <div className="lighttable-grid-area">
        {loading && images.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="lighttable-loading">Loading...</p>
          </div>
        ) : (
          <div
            ref={gridRef}
            className="grid gap-0.5"
            style={{
              gridTemplateColumns: `repeat(auto-fill, minmax(${thumbnailSize}px, 1fr))`,
            }}
          >
            {displayedImages.map((img) => (
              <ThumbnailCard
                key={img.id}
                imgid={img.id}
                filename={img.filename}
                selected={selectedIds.has(img.id)}
                rating={img.rating ?? 0}
                colorLabels={img.color_labels ?? 0}
                groupId={img.group_id ?? 0}
                groupSize={groupSizes.get(img.group_id ?? img.id) ?? 1}
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
