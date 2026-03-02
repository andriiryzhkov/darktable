import { useEffect, useState, useCallback } from "react";
import { useConnectionStore } from "./stores/connectionStore";
import { useCatalogStore } from "./stores/catalogStore";
import LighttableView from "./components/Lighttable/LighttableView";
import DarkroomView from "./components/Darkroom/DarkroomView";

type View = "lighttable" | "darkroom";

function App() {
  const [view, setView] = useState<View>("lighttable");
  const [activeImgId, setActiveImgId] = useState<number | null>(null);
  const { status, error, connect } = useConnectionStore();
  const fetchPage = useCatalogStore((s) => s.fetchPage);

  useEffect(() => {
    connect().then(() => fetchPage(0));
  }, [connect, fetchPage]);

  const openDarkroom = useCallback((imgid: number) => {
    setActiveImgId(imgid);
    setView("darkroom");
  }, []);

  const backToLighttable = useCallback(() => {
    setView("lighttable");
    setActiveImgId(null);
  }, []);

  if (status === "connecting") {
    return (
      <div className="flex items-center justify-center h-screen">
        <p className="text-[var(--text-secondary)]">
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
          <p className="text-[var(--text-secondary)] text-sm">{error}</p>
          <button
            onClick={() => connect()}
            className="mt-4 px-4 py-2 bg-[var(--accent)] rounded text-sm hover:bg-[var(--accent-hover)]"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <>
      {view === "lighttable" && <LighttableView onOpenImage={openDarkroom} />}
      {view === "darkroom" && activeImgId !== null && (
        <DarkroomView imgid={activeImgId} onBack={backToLighttable} />
      )}
    </>
  );
}

export default App;
