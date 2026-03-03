import { useEffect, useState, useRef } from "react";
import { catalogGetThumbnail } from "../../api/commands";
import StarRating from "./StarRating";
import ColorLabels from "./ColorLabels";
import { Ban, Copy, Layers } from "lucide-react";

function RejectIcon({ rejected }: { rejected: boolean }) {
  return (
    <Ban
      size={14}
      style={{
        color: rejected ? "var(--colorlabel-red)" : "var(--thumbnail-font-color)",
        opacity: rejected ? 1 : 0.6,
        cursor: "pointer",
        flexShrink: 0,
      }}
    />
  );
}

interface ThumbnailCardProps {
  imgid: number;
  filename: string;
  selected: boolean;
  rating: number;
  colorLabels: number;
  groupId: number;
  localCopy: boolean;
  altered: boolean;
  onSelect: (e: React.MouseEvent) => void;
  onDoubleClick: () => void;
}

export default function ThumbnailCard({
  imgid,
  filename,
  selected,
  rating,
  colorLabels,
  groupId,
  localCopy,
  altered,
  onSelect,
  onDoubleClick,
}: ThumbnailCardProps) {
  const [src, setSrc] = useState<string | null>(null);
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Observe visibility — load thumbnail only when card enters viewport
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
      { rootMargin: "200px" }, // preload slightly before visible
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Fetch thumbnail JPEG only once visible
  useEffect(() => {
    if (!visible) return;
    let cancelled = false;
    catalogGetThumbnail(imgid)
      .then((result) => {
        if (!cancelled) {
          setSrc(`data:image/jpeg;base64,${result.data}`);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [visible, imgid]);

  // Extract file extension
  const dotIdx = filename.lastIndexOf(".");
  const ext = dotIdx > 0 ? filename.substring(dotIdx + 1) : "";

  return (
    <div
      ref={ref}
      className="thumb-main"
      data-selected={selected}
      onClick={onSelect}
      onDoubleClick={onDoubleClick}
    >
      <div className="thumb-back">
        {/* File extension badge — visible on hover */}
        {ext && <span className="thumb-ext">{ext}</span>}

        {/* Top-right status icons — visible on hover */}
        <div className="thumb-top-right">
          {altered && (
            <span style={{ color: "var(--thumbnail-font-color)", fontSize: 10 }}>
              <Layers size={10} />
            </span>
          )}
          {localCopy && (
            <span style={{ color: "var(--thumbnail-font-color)", fontSize: 10 }}>
              <Copy size={10} />
            </span>
          )}
          {groupId > 0 && (
            <span style={{ color: "var(--thumbnail-font-color)", fontSize: 10 }}>
              G
            </span>
          )}
        </div>

        {/* Image — fits square via CSS */}
        {src ? (
          <img src={src} alt={filename} draggable={false} />
        ) : (
          <div
            className="absolute inset-0 flex items-center justify-center"
          >
            {visible && (
              <div
                className="w-6 h-6 border-2 border-t-transparent rounded-full animate-spin"
                style={{
                  borderColor: "var(--plugin-label-color)",
                  borderTopColor: "transparent",
                }}
              />
            )}
          </div>
        )}

        {/* Bottom overlay: reject left, stars center, color labels right */}
        <div className="thumb-bottom">
          <RejectIcon rejected={rating === 6} />
          <StarRating rating={rating} />
          <ColorLabels labels={colorLabels} />
        </div>
      </div>
    </div>
  );
}
