import LibModuleCard from "../LibModuleCard";

export default function GeotaggingModule() {
  return (
    <LibModuleCard title="geotagging" description="set geolocation information for the currently selected images">
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        manage GPS coordinates
      </p>
    </LibModuleCard>
  );
}
