import { useMemo } from "react";
import LibModuleCard from "../LibModuleCard";
import BauhausSection from "../../controls/BauhausSection";
import { useCatalogStore } from "../../../stores/catalogStore";
import type { ImageInfo } from "../../../types/protocol";

const VARIOUS = "<various values>";

function formatExposure(exposure: number): string {
  if (exposure >= 1) return `${exposure.toFixed(1)}s`;
  return `1/${Math.round(1 / exposure)}s`;
}

function formatDateTime(dt: string | number): string {
  if (!dt) return "-";
  const s = String(dt);
  // Server sends "2024-03-05 14:30:00" from SQLite datetime(), or legacy "2024:03:05 14:30:00"
  if (/^\d{4}-\d{2}-/.test(s)) return s;
  if (/^\d{4}:\d{2}:/.test(s)) return s.replace(/^(\d{4}):(\d{2}):/, "$1-$2-");
  return "-";
}

function formatFileType(flags: number): string {
  if (flags & 64) return "RAW";
  if (flags & 128) return "HDR";
  if (flags & 32) return "JPEG";
  return "";
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

function formatFileSize(bytes: number | undefined): string {
  if (!bytes || bytes <= 0) return "-";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function ExifCell({ value }: { label: string; value: string }) {
  return (
    <div className="exif-cell">
      <span className="exif-cell-value">{value}</span>
    </div>
  );
}

function shortenWB(wb: string | undefined): string {
  if (!wb) return "-";
  const map: Record<string, string> = {
    "auto": "AWB", "Auto": "AWB",
    "daylight": "\u2600", "Daylight": "\u2600",
    "cloudy": "\u2601", "Cloudy": "\u2601",
    "shade": "\u26c5", "Shade": "\u26c5",
    "tungsten": "\ud83d\udca1", "Tungsten": "\ud83d\udca1",
    "fluorescent": "FL", "Fluorescent": "FL",
    "flash": "\u26a1", "Flash": "\u26a1",
  };
  for (const [key, short] of Object.entries(map)) {
    if (wb.toLowerCase().includes(key.toLowerCase())) return short;
  }
  return wb.length > 6 ? wb.slice(0, 5) + "\u2026" : wb;
}

function MeteringIcon({ mode }: { mode: string | undefined }) {
  const s = 14;
  const c = s / 2;
  const stroke = "currentColor";
  const fill = "currentColor";

  if (!mode) return <span className="exif-cell-value">-</span>;
  const m = mode.toLowerCase();

  if (m.includes("spot")) return (
    <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
      <circle cx={c} cy={c} r={5.5} fill="none" stroke={stroke} strokeWidth={1} />
      <circle cx={c} cy={c} r={1.5} fill={fill} />
    </svg>
  );

  if (m.includes("center")) return (
    <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
      <circle cx={c} cy={c} r={5.5} fill="none" stroke={stroke} strokeWidth={1} />
      <circle cx={c} cy={c} r={3} fill={fill} opacity={0.5} />
      <circle cx={c} cy={c} r={1.5} fill={fill} />
    </svg>
  );

  if (m.includes("matrix") || m.includes("evaluative") || m.includes("multi")) return (
    <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
      {[1, 5.5, 10].map(y =>
        [1, 5.5, 10].map(x =>
          <rect key={`${x}-${y}`} x={x} y={y} width={3} height={3} rx={0.5} fill={fill} opacity={0.6} />
        )
      )}
    </svg>
  );

  if (m.includes("partial")) return (
    <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
      <circle cx={c} cy={c} r={5.5} fill="none" stroke={stroke} strokeWidth={1} />
      <circle cx={c} cy={c} r={3} fill={fill} opacity={0.4} />
    </svg>
  );

  if (m.includes("average")) return (
    <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
      <circle cx={c} cy={c} r={5.5} fill={fill} opacity={0.3} />
      <circle cx={c} cy={c} r={5.5} fill="none" stroke={stroke} strokeWidth={1} />
    </svg>
  );

  const text = mode.length > 6 ? mode.slice(0, 5) + "\u2026" : mode;
  return <span className="exif-cell-value">{text}</span>;
}

/** Given selected images, compute set of field names that have different values. */
function computeVarious(imgs: ImageInfo[]): Set<string> {
  const v = new Set<string>();
  if (imgs.length < 2) return v;
  const ref = imgs[0];
  const check = (field: string, get: (i: ImageInfo) => unknown) => {
    const r = get(ref);
    for (let k = 1; k < imgs.length; k++) {
      if (get(imgs[k]) !== r) { v.add(field); return; }
    }
  };
  check("filename", (i) => i.filename);
  check("folder", (i) => i.folder);
  check("film_id", (i) => i.film_id);
  check("id", (i) => i.id);
  check("group_id", (i) => i.group_id);
  check("version", (i) => i.version);
  check("local_copy", (i) => i.local_copy);
  check("import_timestamp", (i) => i.import_timestamp);
  check("change_timestamp", (i) => i.change_timestamp);
  check("export_timestamp", (i) => i.export_timestamp);
  check("print_timestamp", (i) => i.print_timestamp);
  check("maker", (i) => i.maker);
  check("model", (i) => i.model);
  check("lens", (i) => i.lens);
  check("aperture", (i) => i.aperture);
  check("exposure", (i) => i.exposure);
  check("exposure_bias", (i) => i.exposure_bias);
  check("exposure_program", (i) => i.exposure_program);
  check("whitebalance", (i) => i.whitebalance);
  check("flash", (i) => i.flash);
  check("metering_mode", (i) => i.metering_mode);
  check("focal_length", (i) => i.focal_length);
  check("crop", (i) => i.crop);
  check("focus_distance", (i) => i.focus_distance);
  check("iso", (i) => i.iso);
  check("datetime_taken", (i) => i.datetime_taken);
  check("width", (i) => i.width);
  check("height", (i) => i.height);
  check("output_width", (i) => i.output_width);
  check("output_height", (i) => i.output_height);
  check("latitude", (i) => i.latitude);
  check("longitude", (i) => i.longitude);
  check("altitude", (i) => i.altitude);
  check("rating", (i) => i.rating);
  check("color_labels", (i) => i.color_labels);
  check("altered", (i) => i.altered);
  check("flags", (i) => i.flags);
  check("file_size", (i) => i.file_size);
  check("resolution", (i) => `${i.width}x${i.height}`);
  check("camera", (i) => i.model || i.maker);
  return v;
}

function ExifBar({ image, various }: { image: ImageInfo; various: Set<string> }) {
  const v = (field: string, normal: string) => various.has(field) ? "*" : normal;

  const camera = image.model || image.maker || "-";
  const fileType = formatFileType(image.flags) || "-";

  return (
    <div className="exif-summary">
      <div className="exif-filename" style={various.has("filename") ? { fontStyle: "italic" } : undefined}>
        {various.has("filename") ? VARIOUS : image.filename}
      </div>
      <div className="exif-datetime" style={various.has("datetime_taken") ? { fontStyle: "italic" } : undefined}>
        {various.has("datetime_taken") ? VARIOUS : formatDateTime(image.datetime_taken)}
      </div>
      <div className="exif-box">
        {/* Row 1: camera, WB, photometry */}
        <div className="exif-row exif-row-3 exif-row-camera">
          <ExifCell label="camera" value={v("camera", camera)} />
          <ExifCell label="WB" value={v("whitebalance", shortenWB(image.whitebalance))} />
          <div className="exif-cell">
            {various.has("metering_mode")
              ? <span className="exif-cell-value">*</span>
              : <MeteringIcon mode={image.metering_mode} />}
          </div>
        </div>
        {/* Row 2: lens */}
        <div className="exif-row exif-row-1 exif-row-lens">
          <ExifCell label="lens" value={v("lens", image.lens || "-")} />
        </div>
        {/* Row 3: resolution, size, type */}
        <div className="exif-row exif-row-3 exif-row-info">
          <ExifCell label="resolution" value={v("resolution", image.width > 0 && image.height > 0 ? `${image.width}\u00d7${image.height}` : "-")} />
          <ExifCell label="size" value={v("file_size", formatFileSize(image.file_size))} />
          <div className="exif-cell">
            {various.has("flags")
              ? <span className="exif-cell-value">*</span>
              : fileType !== "-" ? <span className="exif-type-badge">{fileType}</span> : <span className="exif-cell-value">-</span>}
          </div>
        </div>
        <div className="exif-divider" />
        <div className="exif-shooting-line">
          <span>{v("iso", image.iso > 0 ? `ISO ${image.iso}` : "-")}</span>
          <span>{v("focal_length", image.focal_length > 0 ? `${image.focal_length.toFixed(0)} mm` : "-")}</span>
          <span>{v("exposure_bias", image.exposure_bias !== undefined && Math.abs(image.exposure_bias) < 1e10 ? `${image.exposure_bias >= 0 ? "+" : ""}${image.exposure_bias.toFixed(1)} EV` : "-")}</span>
          <span>{v("aperture", image.aperture > 0 ? `f/${image.aperture.toFixed(1)}` : "-")}</span>
          <span>{v("exposure", image.exposure > 0 ? formatExposure(image.exposure) : "-")}</span>
        </div>
      </div>
    </div>
  );
}

function InfoRow({ label, value, various }: { label: string; value: string; various?: boolean }) {
  return (
    <div className="info-row">
      <span className="info-row-label">{label}</span>
      <span className="info-row-value" style={various ? { fontStyle: "italic" } : undefined}>
        {various ? VARIOUS : value}
      </span>
    </div>
  );
}

function val(v: number | undefined, fmt: (n: number) => string): string {
  return v !== undefined && v > 0 ? fmt(v) : "-";
}

function formatTimestamp(ts: number | undefined): string {
  if (!ts || ts <= 0) return "-";
  const d = new Date(ts * 1000);
  if (isNaN(d.getTime())) return "-";
  return d.toISOString().replace("T", " ").slice(0, 19);
}

function formatCoord(v: number | undefined): string {
  if (v === undefined || v === 0) return "-";
  return v.toFixed(6);
}

function formatFocalLength35(fl: number, crop: number): string {
  if (fl <= 0 || crop <= 0) return "-";
  return `${(fl * crop).toFixed(0)} mm`;
}

function FullDetails({ image, various }: { image: ImageInfo; various: Set<string> }) {
  const vr = (field: string) => various.has(field);
  return (
    <div className="info-rows">
      {/* darktable internals */}
      <InfoRow label="image ID" value={String(image.id)} various={vr("id")} />
      <InfoRow label="group ID" value={image.group_id !== undefined ? String(image.group_id) : "-"} various={vr("group_id")} />
      <InfoRow label="film roll" value={String(image.film_id)} various={vr("film_id")} />
      <InfoRow label="filename" value={image.filename} various={vr("filename")} />
      <InfoRow label="version" value={image.version !== undefined ? `${image.version} / ${image.max_version}` : "-"} various={vr("version")} />
      <InfoRow label="full path" value={`${image.folder}/${image.filename}`} various={vr("filename") || vr("folder")} />
      <InfoRow label="local copy" value={image.local_copy !== undefined ? (image.local_copy ? "yes" : "no") : "-"} various={vr("local_copy")} />
      <InfoRow label="import time" value={formatTimestamp(image.import_timestamp)} various={vr("import_timestamp")} />
      <InfoRow label="change time" value={formatTimestamp(image.change_timestamp)} various={vr("change_timestamp")} />
      <InfoRow label="export time" value={formatTimestamp(image.export_timestamp)} various={vr("export_timestamp")} />
      <InfoRow label="print time" value={formatTimestamp(image.print_timestamp)} various={vr("print_timestamp")} />

      {/* EXIF */}
      <InfoRow label="maker" value={image.maker || "-"} various={vr("maker")} />
      <InfoRow label="model" value={image.model || "-"} various={vr("model")} />
      <InfoRow label="lens" value={image.lens || "-"} various={vr("lens")} />
      <InfoRow label="aperture" value={val(image.aperture, (v) => `f/${v.toFixed(1)}`)} various={vr("aperture")} />
      <InfoRow label="exposure" value={val(image.exposure, formatExposure)} various={vr("exposure")} />
      <InfoRow label="exposure bias" value={image.exposure_bias !== undefined && image.exposure_bias !== 0 && Math.abs(image.exposure_bias) < 1e10 ? `${image.exposure_bias.toFixed(2)} EV` : "-"} various={vr("exposure_bias")} />
      <InfoRow label="exposure program" value={image.exposure_program || "-"} various={vr("exposure_program")} />
      <InfoRow label="white balance" value={image.whitebalance || "-"} various={vr("whitebalance")} />
      <InfoRow label="flash" value={image.flash || "-"} various={vr("flash")} />
      <InfoRow label="metering mode" value={image.metering_mode || "-"} various={vr("metering_mode")} />
      <InfoRow label="focal length" value={val(image.focal_length, (v) => `${v.toFixed(0)} mm`)} various={vr("focal_length")} />
      <InfoRow label="focal length (35mm)" value={formatFocalLength35(image.focal_length, image.crop ?? 0)} various={vr("focal_length") || vr("crop")} />
      <InfoRow label="crop factor" value={image.crop !== undefined && image.crop > 0 ? image.crop.toFixed(2) : "-"} various={vr("crop")} />
      <InfoRow label="focus distance" value={image.focus_distance !== undefined && image.focus_distance > 0 ? `${image.focus_distance.toFixed(2)} m` : "-"} various={vr("focus_distance")} />
      <InfoRow label="ISO" value={val(image.iso, (v) => String(v))} various={vr("iso")} />
      <InfoRow label="datetime" value={formatDateTime(image.datetime_taken)} various={vr("datetime_taken")} />

      {/* dimensions */}
      <InfoRow label="width" value={image.width > 0 ? String(image.width) : "-"} various={vr("width")} />
      <InfoRow label="height" value={image.height > 0 ? String(image.height) : "-"} various={vr("height")} />
      <InfoRow label="export width" value={image.output_width ? String(image.output_width) : "-"} various={vr("output_width")} />
      <InfoRow label="export height" value={image.output_height ? String(image.output_height) : "-"} various={vr("output_height")} />

      {/* geolocation */}
      <InfoRow label="latitude" value={formatCoord(image.latitude)} various={vr("latitude")} />
      <InfoRow label="longitude" value={formatCoord(image.longitude)} various={vr("longitude")} />
      <InfoRow label="elevation" value={image.altitude !== undefined && image.altitude !== 0 ? `${image.altitude.toFixed(1)} m` : "-"} various={vr("altitude")} />

      {/* tags & metadata */}
      <InfoRow label="tags" value="-" />
      <InfoRow label="categories" value="-" />
      <InfoRow label="rating" value={formatRating(image.rating ?? 0)} various={vr("rating")} />
      <InfoRow label="color labels" value={formatColorLabels(image.color_labels ?? 0)} various={vr("color_labels")} />
      <InfoRow label="altered" value={image.altered !== undefined ? (image.altered ? "yes" : "no") : "-"} various={vr("altered")} />
    </div>
  );
}

const EMPTY_IMAGE: ImageInfo = {
  id: 0, film_id: 0, filename: "-", folder: "-",
  width: 0, height: 0, flags: 0, rating: 0,
  iso: 0, aperture: 0, exposure: 0, focal_length: 0,
  maker: "", model: "", lens: "", datetime_taken: "",
  aspect_ratio: 0,
};

const EMPTY_VARIOUS = new Set<string>();

export default function ImageInfoModule() {
  const selectedIds = useCatalogStore((s) => s.selectedIds);
  const images = useCatalogStore((s) => s.images);
  const hoverImageId = useCatalogStore((s) => s.hoverImageId);

  // Priority: hover image > first selected image > empty
  const hoverImage = hoverImageId ? images.find((img) => img.id === hoverImageId) : null;

  const selectedImages = useMemo(() => {
    if (selectedIds.size === 0) return [];
    return images.filter((img) => selectedIds.has(img.id));
  }, [images, selectedIds]);

  const image = hoverImage ?? selectedImages[0] ?? EMPTY_IMAGE;

  // "various values" only when multiple selected and NOT hovering over a specific image
  const various = useMemo(() => {
    if (hoverImage || selectedImages.length < 2) return EMPTY_VARIOUS;
    return computeVarious(selectedImages);
  }, [hoverImage, selectedImages]);

  return (
    <LibModuleCard title="image information" description="display camera and image metadata for the selected image">
      <ExifBar image={image} various={various} />
      <BauhausSection title="full information">
        <FullDetails image={image} various={various} />
      </BauhausSection>
    </LibModuleCard>
  );
}
