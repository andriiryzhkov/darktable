import NavigationModule from "./modules/NavigationModule";
import ImageInfoModule from "../Sidebar/modules/ImageInfoModule";
import DarkroomHistoryModule from "./modules/DarkroomHistoryModule";

export default function DarkroomLeftSidebar() {
  return (
    <>
      <NavigationModule />
      <ImageInfoModule />
      <DarkroomHistoryModule />
    </>
  );
}
