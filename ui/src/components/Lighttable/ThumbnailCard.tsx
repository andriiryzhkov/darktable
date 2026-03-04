import { useEffect, useState, useRef } from "react";
import { catalogGetThumbnail } from "../../api/commands";
import StarRating from "./StarRating";
import ColorLabels from "./ColorLabels";
import { Ban, Copy, Group, Pencil } from "lucide-react";

function RejectIcon({ rejected }: { rejected: boolean }) {
  return (
    <Ban
      size={14}
      className="thumb-status-icon"
      style={{
        color: rejected ? "var(--colorlabel-red)" : undefined,
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
  groupSize: number;
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
  groupId: _groupId,
  groupSize,
  localCopy,
  altered,
  onSelect,
  onDoubleClick,
}: ThumbnailCardProps) {
  const [src, setSrc] = useState<string | null>(null);
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

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
        {ext && <span className="thumb-ext">{ext}</span>}

        <div className="thumb-top-right">
          {altered && (
            <span className="thumb-status-icon">
              <Pencil size={12} />
            </span>
          )}
          {localCopy && (
            <span className="thumb-status-icon">
              <Copy size={12} />
            </span>
          )}
          {groupSize > 1 && (
            <span className="thumb-status-icon thumb-group-badge">
              <Group size={12} />
              <span className="thumb-group-count">{groupSize}</span>
            </span>
          )}
        </div>

        {src ? (
          <img src={src} alt={filename} draggable={false} />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            {visible && <div className="thumb-spinner" />}
          </div>
        )}

        <div className="thumb-bottom">
          <RejectIcon rejected={rating === 6} />
          <StarRating rating={rating} />
          <ColorLabels labels={colorLabels} />
        </div>
      </div>
    </div>
  );
}
