import { useEffect, useState, useRef, useCallback } from "react";
import { requestThumbnail } from "../../api/thumbnailBatch";
import { useCatalogStore } from "../../stores/catalogStore";
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

  // Load thumbnail when visible
  useEffect(() => {
    if (!visible) return;
    let cancelled = false;
    requestThumbnail(imgid)
      .then((dataUrl) => {
        if (!cancelled) setSrc(dataUrl);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [visible, imgid, thumbRevision]);

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
