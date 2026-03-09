import { Ban, Group, PencilRuler } from "lucide-react";
import StarRating from "../Lighttable/StarRating";
import ColorLabels from "../Lighttable/ColorLabels";

interface Props {
  filename: string;
  ext: string;
  rating: number;
  colorLabels: number;
  groupSize: number;
  localCopy: boolean;
  altered: boolean;
  exposure: number;
  aperture: number;
  iso: number;
  focalLength: number;
  showNormal: boolean;
  showExtended: boolean;
  /** Full-coverage block overlay (HoverBlock mode) */
  showBlock: boolean;
}

export function formatExif(exposure: number, aperture: number, focalLength: number, iso: number): string {
  const parts: string[] = [];
  if (exposure > 0) {
    parts.push(exposure >= 1 ? `${exposure}s` : `1/${Math.round(1 / exposure)}`);
  }
  if (aperture > 0) parts.push(`f/${parseFloat(aperture.toFixed(1))}`);
  if (focalLength > 0) parts.push(`${parseFloat(focalLength.toFixed(1))}mm`);
  if (iso > 0) parts.push(`ISO ${Math.round(iso)}`);
  return parts.join(" \u2022 ");
}

export default function ThumbnailOverlay({
  filename,
  ext,
  rating,
  colorLabels,
  groupSize,
  localCopy,
  altered,
  exposure,
  aperture,
  iso,
  focalLength,
  showNormal,
  showExtended,
  showBlock,
}: Props) {
  // HoverBlock: full-coverage overlay with all info
  if (showBlock) {
    return (
      <div className="thumb-block-overlay">
        <div className="thumb-block-content">
          <div className="thumb-info-filename">{filename}</div>
          <div className="thumb-info-exif">{formatExif(exposure, aperture, focalLength, iso)}</div>
          {altered && (
            <div className="thumb-block-edited">
              <PencilRuler size={12} /> edited
            </div>
          )}
          <div className="thumb-block-rating">
            <Ban
              size={14}
              className="thumb-status-icon"
              style={{
                color: rating === 6 ? "var(--colorlabel-red)" : undefined,
                opacity: rating === 6 ? 1 : 0.6,
                cursor: "pointer",
              }}
            />
            <StarRating rating={rating} />
            <ColorLabels labels={colorLabels} />
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Top-left: file extension */}
      <span
        className="thumb-ext"
        style={{ opacity: showNormal ? 1 : undefined }}
      >
        {ext}
      </span>

      {/* Top-right: status icons */}
      <div
        className="thumb-top-right"
        style={{ opacity: showNormal ? 1 : undefined }}
      >
        {altered && (
          <span className="thumb-status-icon">
            <PencilRuler size={12} />
          </span>
        )}
        {localCopy && (
          <span className="thumb-local-copy" title="local copy" />
        )}
        {groupSize > 1 && (
          <span className="thumb-status-icon thumb-group-badge">
            <Group size={12} />
            <span className="thumb-group-count">{groupSize}</span>
          </span>
        )}
      </div>

      {/* Bottom: extended info + reject, stars, color labels */}
      <div
        className="thumb-bottom"
        style={{ opacity: showNormal ? 1 : undefined }}
      >
        {showExtended && (
          <>
            <div className="thumb-info-filename">{filename}</div>
            <div className="thumb-info-exif">{formatExif(exposure, aperture, focalLength, iso)}</div>
          </>
        )}
        <div className="thumb-bottom-rating">
          <Ban
            size={14}
            className="thumb-status-icon"
            style={{
              color: rating === 6 ? "var(--colorlabel-red)" : undefined,
              opacity: rating === 6 ? 1 : 0.6,
              cursor: "pointer",
              flexShrink: 0,
            }}
          />
          <StarRating rating={rating} />
          <ColorLabels labels={colorLabels} />
        </div>
      </div>
    </>
  );
}
