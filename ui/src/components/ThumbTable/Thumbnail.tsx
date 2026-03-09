import { useEffect, useState, useRef, useCallback } from "react";
import { requestThumbnail } from "../../api/thumbnailBatch";
import { useCatalogStore } from "../../stores/catalogStore";
import { useDevelopStore } from "../../stores/developStore";
import { OverlayMode } from "./types";
import ThumbnailOverlay, { formatExif } from "./ThumbnailOverlay";

interface ThumbnailProps {
  imgid: number;
  filename: string;
  datetimeTaken: string;
  selected: boolean;
  rating: number;
  colorLabels: number;
  groupSize: number;
  localCopy: boolean;
  altered: boolean;
  exposure: number;
  aperture: number;
  iso: number;
  focalLength: number;
  /** Currently being processed in darkroom */
  processing?: boolean;
  overlay: OverlayMode;
  blockTimeout: number;
  showTooltip: boolean;
  onClick: (e: React.MouseEvent) => void;
  onDoubleClick: () => void;
  onHover: (hovering: boolean) => void;
}

export default function Thumbnail({
  imgid,
  filename,
  datetimeTaken,
  selected,
  rating,
  colorLabels,
  groupSize,
  localCopy,
  altered,
  exposure,
  aperture,
  iso,
  focalLength,
  processing,
  overlay,
  blockTimeout,
  showTooltip,
  onClick,
  onDoubleClick,
  onHover,
}: ThumbnailProps) {
  const [src, setSrc] = useState<string | null>(null);
  const [visible, setVisible] = useState(false);
  const [hovering, setHovering] = useState(false);
  const [blockVisible, setBlockVisible] = useState(false);
  const blockTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const thumbRevision = useCatalogStore((s) => s.thumbRevision);

  // Live preview: use develop store's already-fetched preview data
  const devPreviewSrc = useDevelopStore((s) => processing ? s.previewSrc : null);
  const devFrameData = useDevelopStore((s) => processing ? s.frameData : null);
  const devPreviewWidth = useDevelopStore((s) => processing ? s.previewWidth : 0);
  const devPreviewHeight = useDevelopStore((s) => processing ? s.previewHeight : 0);

  // Intersection observer for lazy loading
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: "200px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Load thumbnail when visible (mipmap cache)
  useEffect(() => {
    if (!visible || processing) return;
    let cancelled = false;
    requestThumbnail(imgid)
      .then((dataUrl) => {
        if (!cancelled) setSrc(dataUrl);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [visible, imgid, thumbRevision, processing]);

  // Live preview: derive filmstrip thumbnail from develop store's preview data.
  // previewSrc (JPEG URL) is used directly. frameData (raw BGRA) is rendered
  // to a small offscreen canvas and exported as a blob URL.
  useEffect(() => {
    if (!processing) return;

    // Path 1: JPEG preview URL from develop store — use directly
    if (devPreviewSrc) {
      setSrc(devPreviewSrc);
      return;
    }

    // Path 2: raw BGRA pixels — render to offscreen canvas, export as blob
    if (!devFrameData || !devPreviewWidth || !devPreviewHeight) return;
    let cancelled = false;

    // Scale down to thumbnail size for efficiency
    const maxThumbW = 300;
    const scale = Math.min(1, maxThumbW / devPreviewWidth);
    const tw = Math.round(devPreviewWidth * scale);
    const th = Math.round(devPreviewHeight * scale);

    const canvas = new OffscreenCanvas(tw, th);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Create full-size ImageData from BGRA, then draw scaled
    const fullCanvas = new OffscreenCanvas(devPreviewWidth, devPreviewHeight);
    const fullCtx = fullCanvas.getContext("2d");
    if (!fullCtx) return;
    const imgData = fullCtx.createImageData(devPreviewWidth, devPreviewHeight);
    const src = devFrameData;
    const dst = imgData.data;
    // BGRA → RGBA swap
    for (let i = 0, len = src.length; i < len; i += 4) {
      dst[i] = src[i + 2];     // R ← B
      dst[i + 1] = src[i + 1]; // G
      dst[i + 2] = src[i];     // B ← R
      dst[i + 3] = 255;        // A
    }
    fullCtx.putImageData(imgData, 0, 0);

    // Draw scaled into thumbnail canvas
    ctx.drawImage(fullCanvas, 0, 0, tw, th);

    canvas.convertToBlob({ type: "image/jpeg", quality: 0.7 }).then((blob) => {
      if (cancelled) return;
      const url = URL.createObjectURL(blob);
      setSrc((prev) => {
        if (prev?.startsWith("blob:")) URL.revokeObjectURL(prev);
        return url;
      });
    });

    return () => { cancelled = true; };
  }, [processing, devPreviewSrc, devFrameData, devPreviewWidth, devPreviewHeight]);

  // Handle hover block timer (-1 = stay until mouse leaves, 0 = instant hide, >0 = seconds)
  const handleMouseEnter = useCallback(() => {
    setHovering(true);
    onHover(true);
    if (overlay === OverlayMode.HoverBlock) {
      setBlockVisible(true);
      if (blockTimerRef.current) clearTimeout(blockTimerRef.current);
      if (blockTimeout >= 0) {
        blockTimerRef.current = setTimeout(() => {
          setBlockVisible(false);
        }, blockTimeout * 1000);
      }
    }
  }, [overlay, blockTimeout, onHover]);

  const handleMouseLeave = useCallback(() => {
    setHovering(false);
    onHover(false);
    if (overlay === OverlayMode.HoverBlock) {
      if (blockTimerRef.current) clearTimeout(blockTimerRef.current);
      setBlockVisible(false);
    }
  }, [overlay, onHover]);

  // Determine if overlays should show
  const showBlock = blockVisible && overlay === OverlayMode.HoverBlock;

  const showNormal =
    overlay === OverlayMode.AlwaysNormal ||
    overlay === OverlayMode.AlwaysExtended ||
    overlay === OverlayMode.Mixed ||
    (hovering && (overlay === OverlayMode.HoverNormal || overlay === OverlayMode.HoverExtended));

  const showExtended =
    overlay === OverlayMode.AlwaysExtended ||
    (hovering && (overlay === OverlayMode.HoverExtended || overlay === OverlayMode.Mixed));

  const dotIdx = filename.lastIndexOf(".");
  const ext = dotIdx > 0 ? filename.substring(dotIdx + 1) : "";

  return (
    <div
      ref={ref}
      className="thumb-main"
      data-selected={selected}
      data-processing={processing}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      title={showTooltip ? `${filename}${localCopy ? " (local copy)" : ""}\n${datetimeTaken}\n${formatExif(exposure, aperture, focalLength, iso)}` : undefined}
    >
      <div className="thumb-back">
        {src ? (
          <img src={src} alt={filename} draggable={false} />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            {visible && <div className="thumb-spinner" />}
          </div>
        )}

        {overlay !== OverlayMode.None && (
          <ThumbnailOverlay
            filename={filename}
            ext={ext}
            rating={rating}
            colorLabels={colorLabels}
            groupSize={groupSize}
            localCopy={localCopy}
            altered={altered}
            exposure={exposure}
            aperture={aperture}
            iso={iso}
            focalLength={focalLength}
            showNormal={showNormal}
            showExtended={showExtended}
            showBlock={showBlock}
          />
        )}
      </div>
    </div>
  );
}
