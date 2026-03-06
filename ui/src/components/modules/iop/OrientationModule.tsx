import { useEffect } from "react";
import { RotateCcw, RotateCw, FlipHorizontal2, FlipVertical2 } from "lucide-react";
import { useDevelopStore } from "../../../stores/developStore";
import { useUIStore } from "../../../stores/uiStore";
import BauhausButton from "../../controls/BauhausButton";
import BauhausCheckbox from "../../controls/BauhausCheckbox";
import BauhausTooltip from "../../controls/BauhausTooltip";

// dt_image_orientation_t bitmask values
const FLIP_Y  = 1; // ORIENTATION_FLIP_Y
const FLIP_X  = 2; // ORIENTATION_FLIP_X
const SWAP_XY = 4; // ORIENTATION_SWAP_XY

function rotateCW(orientation: number): number {
  if (orientation & SWAP_XY)
    orientation ^= FLIP_X;
  else
    orientation ^= FLIP_Y;
  return orientation ^ SWAP_XY;
}

function rotateCCW(orientation: number): number {
  if (orientation & SWAP_XY)
    orientation ^= FLIP_Y;
  else
    orientation ^= FLIP_X;
  return orientation ^ SWAP_XY;
}

function flipH(orientation: number): number {
  if (orientation & SWAP_XY)
    return orientation ^ FLIP_Y;
  else
    return orientation ^ FLIP_X;
}

function flipV(orientation: number): number {
  if (orientation & SWAP_XY)
    return orientation ^ FLIP_X;
  else
    return orientation ^ FLIP_Y;
}

export default function OrientationModule() {
  const flipParams = useDevelopStore((s) => s.flipParams);
  const setModuleParam = useDevelopStore((s) => s.setModuleParam);
  const fetchModuleParams = useDevelopStore((s) => s.fetchModuleParams);
  const showGuides = useUIStore((s) => s.showGuides);
  const setShowGuides = useUIStore((s) => s.setShowGuides);
  const setGuidesModuleOpen = useUIStore((s) => s.setGuidesModuleOpen);

  useEffect(() => {
    if (!flipParams) fetchModuleParams("flip");
  }, [flipParams, fetchModuleParams]);

  useEffect(() => {
    setGuidesModuleOpen(true);
    return () => setGuidesModuleOpen(false);
  }, [setGuidesModuleOpen]);

  if (!flipParams) {
    return (
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        loading orientation params…
      </p>
    );
  }

  // ORIENTATION_NULL (-1) means autodetect; treat as NONE (0) for transforms
  const current = flipParams.orientation < 0 ? 0 : flipParams.orientation;

  const apply = (newOrientation: number) => {
    setModuleParam("flip", { orientation: newOrientation });
  };

  return (
    <>
      <div className="orientation-row">
        <span className="orientation-label">transform</span>
        <div className="orientation-buttons">
          <BauhausTooltip content="rotate 90° CCW">
            <BauhausButton
              icon={<RotateCcw size={14} />}
              transparent
              onClick={() => apply(rotateCCW(current))}
            />
          </BauhausTooltip>
          <BauhausTooltip content="rotate 90° CW">
            <BauhausButton
              icon={<RotateCw size={14} />}
              transparent
              onClick={() => apply(rotateCW(current))}
            />
          </BauhausTooltip>
          <BauhausTooltip content="flip horizontally">
            <BauhausButton
              icon={<FlipHorizontal2 size={14} />}
              transparent
              onClick={() => apply(flipH(current))}
            />
          </BauhausTooltip>
          <BauhausTooltip content="flip vertically">
            <BauhausButton
              icon={<FlipVertical2 size={14} />}
              transparent
              onClick={() => apply(flipV(current))}
            />
          </BauhausTooltip>
        </div>
      </div>
      <BauhausCheckbox
        label="show guides"
        checked={showGuides}
        onChange={setShowGuides}
      />
    </>
  );
}
