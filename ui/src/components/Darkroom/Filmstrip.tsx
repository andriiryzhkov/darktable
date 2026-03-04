import { useRef, useEffect, useState, useCallback } from "react";
import { useCatalogStore } from "../../stores/catalogStore";
import { catalogGetThumbnail } from "../../api/commands";
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
  const scrollRef = useRef<HTMLDivElement>(null);

  // Scroll to selected image on mount
  useEffect(() => {
    if (!scrollRef.current || selectedIds.size === 0) return;
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
      onSelectImage(imgid);
    },
    [selectImage, onSelectImage],
  );

  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (scrollRef.current) {
      scrollRef.current.scrollLeft += e.deltaY;
    }
  }, []);

  const activeId = selectedIds.size > 0 ? [...selectedIds][0] : null;
  const activeImage = activeId
    ? images.find((i) => i.id === activeId)
    : null;

  return (
    <div className="filmstrip" style={{ height: FILMSTRIP_HEIGHT }}>
      <div ref={scrollRef} className="filmstrip-scroll" onWheel={handleWheel}>
        {images.map((img) => (
          <FilmstripThumb
            key={img.id}
            image={img}
            active={selectedIds.has(img.id)}
            onClick={() => handleClick(img.id)}
          />
        ))}
      </div>
      {activeImage && (
        <div className="filmstrip-info">
          <span className="filmstrip-exif">
            {formatExifSummary(activeImage)}
          </span>
        </div>
      )}
    </div>
  );
}

function FilmstripThumb({
  image,
  active,
  onClick,
}: {
  image: ImageInfo;
  active: boolean;
  onClick: () => void;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          catalogGetThumbnail(image.id)
            .then((result) =>
              setSrc(`data:image/jpeg;base64,${result.data}`),
            )
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
      data-active={active}
      onClick={onClick}
      style={{ width: THUMB_WIDTH }}
    >
      {src && <img src={src} alt={image.filename} draggable={false} />}
    </div>
  );
}

function formatExposure(exposure: number): string {
  if (exposure >= 1) return `${exposure.toFixed(1)}s`;
  return `1/${Math.round(1 / exposure)}`;
}

function formatExifSummary(img: ImageInfo): string {
  const parts: string[] = [];
  if (img.exposure > 0) parts.push(formatExposure(img.exposure));
  if (img.aperture > 0) parts.push(`f/${img.aperture.toFixed(1)}`);
  if (img.focal_length > 0)
    parts.push(`${img.focal_length.toFixed(1)} mm`);
  if (img.iso > 0) parts.push(`ISO ${img.iso}`);
  return parts.join(" \u2022 ");
}
