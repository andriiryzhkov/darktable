import { useEffect } from "react";
import { useDevelopStore } from "../../stores/developStore";
import PreviewCanvas from "./PreviewCanvas";
import ExposureModule from "./ExposureModule";

interface Props {
  imgid: number;
  onBack: () => void;
}

export default function DarkroomView({ imgid, onBack }: Props) {
  const { openSession, closeSession, loading } = useDevelopStore();

  useEffect(() => {
    openSession(imgid);
    return () => {
      closeSession();
    };
  }, [imgid, openSession, closeSession]);

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center px-4 py-2 bg-[var(--plugin-bg-color)] border-b border-[var(--border-color)]">
        <button
          onClick={onBack}
          className="px-3 py-1 text-sm bg-[var(--button-bg)] rounded hover:bg-[var(--button-hover-bg)]"
        >
          Back
        </button>
        <span className="ml-4 text-sm text-[var(--plugin-label-color)]">
          Darkroom {loading ? "(processing...)" : ""}
        </span>
      </div>

      {/* Main content: preview + module panel */}
      <div className="flex flex-1 overflow-hidden">
        {/* Preview area */}
        <div className="flex-1 flex items-center justify-center bg-black overflow-hidden">
          <PreviewCanvas />
        </div>

        {/* Module panel */}
        <div className="w-[300px] bg-[var(--plugin-bg-color)] border-l border-[var(--border-color)] overflow-y-auto">
          <ExposureModule />
        </div>
      </div>
    </div>
  );
}
