import CollapsibleModule from "../CollapsibleModule";
import { useCatalogStore } from "../../../stores/catalogStore";
import type { ImageInfo } from "../../../types/protocol";

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="info-row">
      <span className="info-row-label">{label}</span>
      <span className="info-row-value">{value}</span>
    </div>
  );
}

function formatExposure(exposure: number): string {
  if (exposure >= 1) return `${exposure.toFixed(1)}s`;
  return `1/${Math.round(1 / exposure)}s`;
}

function formatDateTime(timestamp: number): string {
  if (!timestamp) return "-";
  const d = new Date(timestamp * 1000);
  const date = d.toLocaleDateString("en-CA");
  const time = d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  return `${date} ${time}`;
}

function formatRating(rating: number): string {
  if (rating === 6) return "rejected";
  if (rating === 0) return "unrated";
  return "\u2605".repeat(rating);
}

const COLOR_NAMES = ["red", "yellow", "green", "blue", "purple"] as const;

function formatColorLabels(mask: number): string {
  const labels: string[] = [];
  for (let i = 0; i < COLOR_NAMES.length; i++) {
    if (mask & (1 << i)) labels.push(COLOR_NAMES[i]);
  }
  return labels.length > 0 ? labels.join(", ") : "none";
}

function val(v: number | undefined, fmt: (n: number) => string): string {
  return v !== undefined && v > 0 ? fmt(v) : "-";
}

function ImageDetails({ image }: { image: ImageInfo }) {
  return (
    <div className="info-rows">
      {/* darktable internals */}
      <InfoRow label="image ID" value={String(image.id)} />
      <InfoRow label="group ID" value={image.group_id !== undefined ? String(image.group_id) : "-"} />
      <InfoRow label="film roll" value={String(image.film_id)} />
      <InfoRow label="filename" value={image.filename} />
      <InfoRow label="version" value="-" />
      <InfoRow label="full path" value={`${image.folder}/${image.filename}`} />
      <InfoRow label="local copy" value={image.local_copy !== undefined ? (image.local_copy ? "yes" : "no") : "-"} />
      <InfoRow label="import time" value="-" />
      <InfoRow label="change time" value="-" />
      <InfoRow label="export time" value="-" />
      <InfoRow label="print time" value="-" />

      {/* EXIF */}
      <InfoRow label="maker" value="-" />
      <InfoRow label="model" value="-" />
      <InfoRow label="lens" value="-" />
      <InfoRow label="aperture" value={val(image.aperture, (v) => `f/${v.toFixed(1)}`)} />
      <InfoRow label="exposure" value={val(image.exposure, formatExposure)} />
      <InfoRow label="exposure bias" value="-" />
      <InfoRow label="exposure program" value="-" />
      <InfoRow label="white balance" value="-" />
      <InfoRow label="flash" value="-" />
      <InfoRow label="metering mode" value="-" />
      <InfoRow label="focal length" value={val(image.focal_length, (v) => `${v.toFixed(0)} mm`)} />
      <InfoRow label="focal length (35mm)" value="-" />
      <InfoRow label="crop factor" value="-" />
      <InfoRow label="focus distance" value="-" />
      <InfoRow label="ISO" value={val(image.iso, (v) => String(v))} />
      <InfoRow label="datetime" value={formatDateTime(image.datetime_taken)} />

      {/* dimensions */}
      <InfoRow label="width" value={String(image.width)} />
      <InfoRow label="height" value={String(image.height)} />
      <InfoRow label="export width" value="-" />
      <InfoRow label="export height" value="-" />

      {/* geolocation */}
      <InfoRow label="latitude" value="-" />
      <InfoRow label="longitude" value="-" />
      <InfoRow label="elevation" value="-" />

      {/* tags & metadata */}
      <InfoRow label="tags" value="-" />
      <InfoRow label="categories" value="-" />
      <InfoRow label="rating" value={formatRating(image.rating ?? 0)} />
      <InfoRow label="color labels" value={formatColorLabels(image.color_labels ?? 0)} />
      <InfoRow label="altered" value={image.altered !== undefined ? (image.altered ? "yes" : "no") : "-"} />
    </div>
  );
}

export default function ImageInfoModule() {
  const selectedIds = useCatalogStore((s) => s.selectedIds);
  const images = useCatalogStore((s) => s.images);

  const firstSelectedId = selectedIds.size > 0 ? [...selectedIds][0] : null;
  const selected = firstSelectedId
    ? images.find((img) => img.id === firstSelectedId)
    : null;

  return (
    <CollapsibleModule title="image information" defaultOpen>
      {selected ? (
        <ImageDetails image={selected} />
      ) : (
        <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
          no image selected
        </p>
      )}
    </CollapsibleModule>
  );
}
