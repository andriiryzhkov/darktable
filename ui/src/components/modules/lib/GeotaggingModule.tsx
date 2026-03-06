import CollapsibleModule from "../CollapsibleModule";

export default function GeotaggingModule() {
  return (
    <CollapsibleModule title="geotagging" description="set geolocation information for the currently selected images">
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        manage GPS coordinates
      </p>
    </CollapsibleModule>
  );
}
