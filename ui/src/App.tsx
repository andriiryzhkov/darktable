import { useEffect, useCallback } from "react";
import { useConnectionStore } from "./stores/connectionStore";
import { useCatalogStore } from "./stores/catalogStore";
import { useUIStore } from "./stores/uiStore";
import HeaderBar from "./components/Layout/HeaderBar";
import Sidebar from "./components/Layout/Sidebar";
import BottomBar from "./components/Layout/BottomBar";
import LeftSidebarModules from "./components/Sidebar/LeftSidebarModules";
import RightSidebarModules from "./components/Sidebar/RightSidebarModules";
import LighttableView from "./components/Lighttable/LighttableView";
import DarkroomView from "./components/Darkroom/DarkroomView";
import DarkroomLeftSidebar from "./components/Darkroom/DarkroomLeftSidebar";
import DarkroomRightSidebar from "./components/Darkroom/DarkroomRightSidebar";
import Filmstrip from "./components/Darkroom/Filmstrip";
import ImportDialog from "./components/Import/ImportDialog";

function App() {
  const { status, error, connect } = useConnectionStore();
  const fetchAll = useCatalogStore((s) => s.fetchAll);
  const {
    activeView,
    setActiveView,
    leftSidebarOpen,
    rightSidebarOpen,
    leftSidebarWidth,
    rightSidebarWidth,
    toggleLeftSidebar,
    toggleRightSidebar,
    setLeftSidebarWidth,
    setRightSidebarWidth,
  } = useUIStore();
  const activeImgId = useCatalogStore((s) => {
    const ids = s.selectedIds;
    return ids.size > 0 ? [...ids][0] : null;
  });

  useEffect(() => {
    connect().then(() => fetchAll());
  }, [connect, fetchAll]);

  const openDarkroom = useCallback(
    (imgid: number) => {
      useCatalogStore.getState().selectImage(imgid);
      setActiveView("darkroom");
    },
    [setActiveView],
  );

  if (status === "connecting") {
    return (
      <div className="flex items-center justify-center h-screen">
        <p style={{ color: "var(--plugin-label-color)" }}>
          Connecting to darktable server...
        </p>
      </div>
    );
  }

  if (status === "disconnected" && error) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <p className="text-red-400 mb-2">Connection failed</p>
          <p className="text-sm mb-4" style={{ color: "var(--plugin-label-color)" }}>
            {error}
          </p>
          <button
            onClick={() => connect()}
            className="px-4 py-2 rounded text-sm"
            style={{
              backgroundColor: "var(--button-bg)",
              color: "var(--button-fg)",
            }}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen">
      <HeaderBar />

      <div className="flex flex-1 overflow-hidden">
        {/* Left sidebar */}
        <Sidebar
          side="left"
          open={leftSidebarOpen}
          width={leftSidebarWidth}
          onToggle={toggleLeftSidebar}
          onResize={setLeftSidebarWidth}
        >
          {activeView === "lighttable" && <LeftSidebarModules />}
          {activeView === "darkroom" && <DarkroomLeftSidebar />}
        </Sidebar>

        {/* Center content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {activeView === "lighttable" && (
            <LighttableView onOpenImage={openDarkroom} />
          )}
          {activeView === "darkroom" && activeImgId !== null && (
            <DarkroomView imgid={activeImgId} />
          )}
          {activeView === "lighttable" && <BottomBar />}
        </div>

        {/* Right sidebar */}
        <Sidebar
          side="right"
          open={rightSidebarOpen}
          width={rightSidebarWidth}
          onToggle={toggleRightSidebar}
          onResize={setRightSidebarWidth}
        >
          {activeView === "lighttable" && <RightSidebarModules />}
          {activeView === "darkroom" && <DarkroomRightSidebar />}
        </Sidebar>
      </div>
      {activeView === "darkroom" && (
        <Filmstrip onSelectImage={openDarkroom} />
      )}
      <ImportDialog />
    </div>
  );
}

export default App;
