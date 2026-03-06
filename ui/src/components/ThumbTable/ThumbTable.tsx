import { useRef, useEffect, useCallback, useMemo } from "react";
import { useCatalogStore } from "../../stores/catalogStore";
import { useFilterStore } from "../../stores/filterStore";
import { useUIStore } from "../../stores/uiStore";
import { useOverlayStore } from "../../stores/overlayStore";
import { ThumbTableMode } from "./types";
import Thumbnail from "./Thumbnail";


const FILMSTRIP_HEIGHT = 100;

interface Props {
  mode: ThumbTableMode;
  /** Image id currently being processed in darkroom (filmstrip only) */
  processingImgId?: number | null;
  onDoubleClickImage: (imgid: number) => void;
}

export default function ThumbTable({
  mode,
  processingImgId,
  onDoubleClickImage,
}: Props) {
  const images = useCatalogStore((s) => s.images);
  const loading = useCatalogStore((s) => s.loading);
  const selectedIds = useCatalogStore((s) => s.selectedIds);
  const selectImage = useCatalogStore((s) => s.selectImage);
  const setHoverImageId = useCatalogStore((s) => s.setHoverImageId);
  const overlayCfg = useOverlayStore((s) => s.modes[mode]);

  const isFilemanager = mode === ThumbTableMode.Filemanager;

  // Filemanager-specific state
  const grouping = useFilterStore((s) => s.grouping);
  const thumbnailSize = useUIStore((s) => s.thumbnailSize);
  const setGridColumns = useUIStore((s) => s.setGridColumns);

  const groupSizes = useMemo(() => {
    if (!isFilemanager) return new Map<number, number>();
    const counts = new Map<number, number>();
    for (const img of images) {
      const gid = img.group_id ?? img.id;
      counts.set(gid, (counts.get(gid) ?? 0) + 1);
    }
    return counts;
  }, [images, isFilemanager]);

  const displayedImages = useMemo(
    () =>
      isFilemanager && grouping
        ? images.filter((img) => img.id === (img.group_id ?? img.id))
        : images,
    [images, grouping, isFilemanager],
  );

  const containerRef = useRef<HTMLDivElement>(null);

  // Filemanager: measure grid columns
  useEffect(() => {
    if (!isFilemanager) return;
    const el = containerRef.current;
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
  }, [isFilemanager, thumbnailSize, setGridColumns]);

  // Filmstrip: scroll to selected image on mount
  const didInitScroll = useRef(false);
  useEffect(() => {
    if (isFilemanager) return;
    if (didInitScroll.current || !containerRef.current || selectedIds.size === 0) return;
    didInitScroll.current = true;
    const activeId = [...selectedIds][0];
    const idx = images.findIndex((img) => img.id === activeId);
    if (idx >= 0 && containerRef.current) {
      const thumbWidth = 110;
      const scrollLeft =
        idx * (thumbWidth + 2) -
        containerRef.current.clientWidth / 2 +
        thumbWidth / 2;
      containerRef.current.scrollLeft = Math.max(0, scrollLeft);
    }
  }, [images, selectedIds, isFilemanager]);

  const handleClick = useCallback(
    (imgid: number, e: React.MouseEvent) => {
      if (isFilemanager) {
        selectImage(imgid, e);
      } else {
        selectImage(imgid);
      }
    },
    [isFilemanager, selectImage],
  );

  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (containerRef.current) {
      containerRef.current.scrollLeft += e.deltaY;
    }
  }, []);

  if (isFilemanager) {
    return (
      <div className="lighttable-grid-area">
        {loading && images.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="lighttable-loading">Loading...</p>
          </div>
        ) : (
          <div
            ref={containerRef}
            className="grid gap-0.5"
            style={{
              gridTemplateColumns: `repeat(auto-fill, minmax(${thumbnailSize}px, 1fr))`,
            }}
          >
            {displayedImages.map((img) => (
              <Thumbnail
                key={img.id}
                imgid={img.id}
                filename={img.filename}
                datetimeTaken={img.datetime_taken}
                selected={selectedIds.has(img.id)}
                rating={img.rating ?? 0}
                colorLabels={img.color_labels ?? 0}
                groupSize={groupSizes.get(img.group_id ?? img.id) ?? 1}
                localCopy={img.local_copy ?? false}
                altered={img.altered ?? false}
                exposure={img.exposure ?? 0}
                aperture={img.aperture ?? 0}
                iso={img.iso ?? 0}
                focalLength={img.focal_length ?? 0}
                overlay={overlayCfg.overlay}
                blockTimeout={overlayCfg.blockTimeout}
                showTooltip={overlayCfg.tooltip}
                onClick={(e) => handleClick(img.id, e)}
                onDoubleClick={() => onDoubleClickImage(img.id)}
                onHover={(h) => setHoverImageId(h ? img.id : null)}
              />
            ))}
          </div>
        )}
      </div>
    );
  }

  // Filmstrip mode
  return (
    <div className="filmstrip" style={{ height: FILMSTRIP_HEIGHT }}>
      <div
        ref={containerRef}
        className="filmstrip-scroll"
        onWheel={handleWheel}
      >
        {images.map((img) => (
          <Thumbnail
            key={img.id}
            imgid={img.id}
            filename={img.filename}
            datetimeTaken={img.datetime_taken}
            selected={selectedIds.has(img.id)}
            rating={img.rating ?? 0}
            colorLabels={img.color_labels ?? 0}
            groupSize={1}
            localCopy={img.local_copy ?? false}
            altered={img.altered ?? false}
            exposure={img.exposure ?? 0}
            aperture={img.aperture ?? 0}
            iso={img.iso ?? 0}
            focalLength={img.focal_length ?? 0}
            processing={img.id === processingImgId}
            overlay={overlayCfg.overlay}
            blockTimeout={overlayCfg.blockTimeout}
            showTooltip={overlayCfg.tooltip}
            onClick={(e) => handleClick(img.id, e)}
            onDoubleClick={() => onDoubleClickImage(img.id)}
            onHover={(h) => setHoverImageId(h ? img.id : null)}
          />
        ))}
      </div>
    </div>
  );
}
