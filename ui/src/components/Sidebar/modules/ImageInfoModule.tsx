import CollapsibleModule from "../CollapsibleModule";
import { useCatalogStore } from "../../../stores/catalogStore";

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span style={{ color: "var(--plugin-label-color)" }}>{label}</span>
      <span style={{ color: "var(--fg-color)" }}>{value}</span>
    </div>
  );
}

export default function ImageInfoModule() {
  const selectedIds = useCatalogStore((s) => s.selectedIds);
  const images = useCatalogStore((s) => s.images);

  // Show info for the first selected image
  const firstSelectedId = selectedIds.size > 0 ? [...selectedIds][0] : null;
  const selected = firstSelectedId
    ? images.find((img) => img.id === firstSelectedId)
    : null;

  return (
    <CollapsibleModule title="image information" defaultOpen>
      {selected ? (
        <div className="text-xs space-y-0.5">
          <InfoRow label="file" value={selected.filename} />
          <InfoRow label="folder" value={selected.folder} />
          <InfoRow
            label="size"
            value={`${selected.width} × ${selected.height}`}
          />
          {selected.exposure > 0 && (
            <InfoRow
              label="exposure"
              value={`1/${Math.round(1 / selected.exposure)}s`}
            />
          )}
          {selected.aperture > 0 && (
            <InfoRow
              label="aperture"
              value={`f/${selected.aperture.toFixed(1)}`}
            />
          )}
          {selected.iso > 0 && (
            <InfoRow label="ISO" value={String(selected.iso)} />
          )}
          {selected.focal_length > 0 && (
            <InfoRow
              label="focal"
              value={`${selected.focal_length.toFixed(0)}mm`}
            />
          )}
        </div>
      ) : (
        <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
          no image selected
        </p>
      )}
    </CollapsibleModule>
  );
}
