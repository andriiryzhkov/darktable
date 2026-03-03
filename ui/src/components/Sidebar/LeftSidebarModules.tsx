import ImportModule from "./modules/ImportModule";
import CollectionsModule from "./modules/CollectionsModule";
import CollectionFiltersModule from "./modules/CollectionFiltersModule";
import ImageInfoModule from "./modules/ImageInfoModule";
import ScriptsModule from "./modules/ScriptsModule";
import ExportModule from "./modules/ExportModule";

export default function LeftSidebarModules() {
  return (
    <>
      <ImportModule />
      <CollectionsModule />
      <CollectionFiltersModule />
      <ImageInfoModule />
      <ScriptsModule />
      <ExportModule />
    </>
  );
}
