import { useCallback, useEffect, useState } from "react";
import { useCatalogStore } from "../../stores/catalogStore";
import { catalogGetThumbnail } from "../../api/commands";

interface Props {
  onOpenImage: (imgid: number) => void;
}

function ThumbnailCard({
  imgid,
  filename,
  selected,
  onSelect,
  onDoubleClick,
}: {
  imgid: number;
  filename: string;
  selected: boolean;
  onSelect: () => void;
  onDoubleClick: () => void;
}) {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    catalogGetThumbnail(imgid).then((result) => {
      if (!cancelled) {
        setSrc(`data:image/jpeg;base64,${result.data}`);
      }
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [imgid]);

  return (
    <div
      className={`cursor-pointer rounded overflow-hidden transition-all ${
        selected
          ? "ring-2 ring-[var(--accent)] bg-[var(--bg-tertiary)]"
          : "hover:bg-[var(--bg-tertiary)]"
      }`}
      onClick={onSelect}
      onDoubleClick={onDoubleClick}
    >
      <div className="aspect-square bg-[var(--bg-secondary)] flex items-center justify-center">
        {src ? (
          <img
            src={src}
            alt={filename}
            className="max-w-full max-h-full object-contain"
          />
        ) : (
          <div className="w-8 h-8 border-2 border-[var(--text-secondary)] border-t-transparent rounded-full animate-spin" />
        )}
      </div>
      <div className="px-2 py-1 text-xs text-[var(--text-secondary)] truncate">
        {filename}
      </div>
    </div>
  );
}

export default function LighttableView({ onOpenImage }: Props) {
  const { images, total, offset, limit, loading, selectedId, selectImage, nextPage, prevPage } =
    useCatalogStore();

  const handleSelect = useCallback(
    (id: number) => selectImage(id),
    [selectImage],
  );

  const pageNum = Math.floor(offset / limit) + 1;
  const totalPages = Math.ceil(total / limit);

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-[var(--bg-secondary)] border-b border-[var(--border)]">
        <span className="text-sm text-[var(--text-secondary)]">
          {total} images
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={prevPage}
            disabled={offset === 0 || loading}
            className="px-3 py-1 text-sm bg-[var(--bg-tertiary)] rounded disabled:opacity-30"
          >
            Prev
          </button>
          <span className="text-sm text-[var(--text-secondary)]">
            {pageNum} / {totalPages || 1}
          </span>
          <button
            onClick={nextPage}
            disabled={offset + limit >= total || loading}
            className="px-3 py-1 text-sm bg-[var(--bg-tertiary)] rounded disabled:opacity-30"
          >
            Next
          </button>
        </div>
      </div>

      {/* Thumbnail grid */}
      <div className="flex-1 overflow-y-auto p-4">
        {loading && images.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-[var(--text-secondary)]">Loading...</p>
          </div>
        ) : (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-3">
            {images.map((img) => (
              <ThumbnailCard
                key={img.id}
                imgid={img.id}
                filename={img.filename}
                selected={selectedId === img.id}
                onSelect={() => handleSelect(img.id)}
                onDoubleClick={() => onOpenImage(img.id)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
