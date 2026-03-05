import { useEffect } from "react";
import { useDevelopStore } from "../../stores/developStore";
import PreviewCanvas from "./PreviewCanvas";
import ImageInfo from "./ImageInfo";
import TopToolbar from "../Lighttable/TopToolbar";

interface Props {
  imgid: number;
}

export default function DarkroomView({ imgid }: Props) {
  const { openSession, closeSession } = useDevelopStore();

  useEffect(() => {
    openSession(imgid);
    return () => {
      closeSession();
    };
  }, [imgid, openSession, closeSession]);

  return (
    <div className="flex flex-col flex-1 min-h-0">
      <TopToolbar />
      <div
        className="flex-1 flex items-center justify-center overflow-hidden"
        style={{ backgroundColor: "var(--darkroom-bg-color)" }}
      >
        <PreviewCanvas />
      </div>
      <ImageInfo />
    </div>
  );
}
