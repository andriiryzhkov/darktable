import { useCatalogStore } from "../../stores/catalogStore";
import type { ImageInfo as ImageInfoType } from "../../types/protocol";

function formatExposure(exposure: number): string {
  if (exposure >= 1) return `${exposure.toFixed(1)}s`;
  return `1/${Math.round(1 / exposure)}`;
}

function formatExifSummary(img: ImageInfoType): string {
  const parts: string[] = [];
  if (img.exposure > 0) parts.push(formatExposure(img.exposure));
  if (img.aperture > 0) parts.push(`f/${img.aperture.toFixed(1)}`);
  if (img.focal_length > 0) parts.push(`${img.focal_length.toFixed(1)} mm`);
  if (img.iso > 0) parts.push(`ISO ${img.iso}`);
  return parts.join(" \u2022 ");
}

export default function ImageInfo() {
  const images = useCatalogStore((s) => s.images);
  const selectedIds = useCatalogStore((s) => s.selectedIds);

  const activeId = selectedIds.size > 0 ? [...selectedIds][0] : null;
  const activeImage = activeId ? images.find((i) => i.id === activeId) : null;

  if (!activeImage) return null;

  return (
    <div className="image-info">
      <span className="image-info-exif">
        {formatExifSummary(activeImage)}
      </span>
    </div>
  );
}
