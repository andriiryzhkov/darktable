import CollapsibleModule from "../CollapsibleModule";

export default function GeotaggingModule() {
  return (
    <CollapsibleModule title="geotagging">
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        manage GPS coordinates
      </p>
    </CollapsibleModule>
  );
}
