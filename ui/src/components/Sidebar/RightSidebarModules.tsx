import TaggingModule from "./modules/TaggingModule";
import StylesModule from "./modules/StylesModule";
import EditMetadataModule from "./modules/EditMetadataModule";
import SelectionModule from "./modules/SelectionModule";
import ActionsModule from "./modules/ActionsModule";
import HistoryStackModule from "./modules/HistoryStackModule";
import GeotaggingModule from "./modules/GeotaggingModule";

export default function RightSidebarModules() {
  return (
    <>
      <TaggingModule />
      <StylesModule />
      <EditMetadataModule />
      <SelectionModule />
      <ActionsModule />
      <HistoryStackModule />
      <GeotaggingModule />
    </>
  );
}
