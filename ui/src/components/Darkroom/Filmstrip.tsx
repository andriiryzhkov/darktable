import { useRef, useEffect, useState, useCallback } from "react";
import { useCatalogStore } from "../../stores/catalogStore";
import { useUIStore } from "../../stores/uiStore";
import { requestThumbnail } from "../../api/thumbnailBatch";
import type { ImageInfo } from "../../types/protocol";

const FILMSTRIP_HEIGHT = 100;
const THUMB_WIDTH = 110;

interface Props {
  onSelectImage: (imgid: number) => void;
}

export default function Filmstrip({ onSelectImage }: Props) {
  const images = useCatalogStore((s) => s.images);
  const selectedIds = useCatalogStore((s) => s.selectedIds);
  const selectImage = useCatalogStore((s) => s.selectImage);
  const setHoverImageId = useCatalogStore((s) => s.setHoverImageId);
  const darkroomImgId = useUIStore((s) => s.darkroomImgId);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Scroll to selected image on mount only
  const didInitScroll = useRef(false);
  useEffect(() => {
    if (didInitScroll.current || !scrollRef.current || selectedIds.size === 0) return;
    didInitScroll.current = true;
    const activeId = [...selectedIds][0];
    const idx = images.findIndex((img) => img.id === activeId);
    if (idx >= 0) {
      const scrollLeft =
        idx * (THUMB_WIDTH + 2) -
        scrollRef.current.clientWidth / 2 +
        THUMB_WIDTH / 2;
      scrollRef.current.scrollLeft = Math.max(0, scrollLeft);
    }
  }, [images, selectedIds]);

  const handleClick = useCallback(
    (imgid: number) => {
      selectImage(imgid);
    },
    [selectImage],
  );

  const handleDoubleClick = useCallback(
    (imgid: number) => {
      onSelectImage(imgid);
    },
    [onSelectImage],
  );

  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (scrollRef.current) {
      scrollRef.current.scrollLeft += e.deltaY;
    }
  }, []);

  return (
    <div className="filmstrip" style={{ height: FILMSTRIP_HEIGHT }}>
      <div ref={scrollRef} className="filmstrip-scroll" onWheel={handleWheel}>
        {images.map((img) => (
          <FilmstripThumb
            key={img.id}
            image={img}
            selected={selectedIds.has(img.id)}
            processing={img.id === darkroomImgId}
            onClick={() => handleClick(img.id)}
            onDoubleClick={() => handleDoubleClick(img.id)}
            onHover={(h) => setHoverImageId(h ? img.id : null)}
          />
        ))}
      </div>
    </div>
  );
}

function FilmstripThumb({
  image,
  selected,
  processing,
  onClick,
  onDoubleClick,
  onHover,
}: {
  image: ImageInfo;
  selected: boolean;
  processing: boolean;
  onClick: () => void;
  onDoubleClick: () => void;
  onHover: (hovering: boolean) => void;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          requestThumbnail(image.id)
            .then((dataUrl) => setSrc(dataUrl))
            .catch(() => {});
          observer.disconnect();
        }
      },
      { rootMargin: "200px", root: el.parentElement },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [image.id]);

  return (
    <div
      ref={ref}
      className="filmstrip-thumb"
      data-selected={selected}
      data-processing={processing}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
      style={{ width: THUMB_WIDTH }}
    >
      {src && <img src={src} alt={image.filename} draggable={false} />}
    </div>
  );
}

