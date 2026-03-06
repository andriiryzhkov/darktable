import {
  Star,
  Plus,
  Minus,
  Ban,
  CircleOff,
} from "lucide-react";
import { useUIStore } from "../../stores/uiStore";
import BauhausButton from "../controls/BauhausButton";
import BauhausTooltip from "../controls/BauhausTooltip";

const COLORS = [
  { key: "red", var: "--colorlabel-red" },
  { key: "yellow", var: "--colorlabel-yellow" },
  { key: "green", var: "--colorlabel-green" },
  { key: "blue", var: "--colorlabel-blue" },
  { key: "purple", var: "--colorlabel-purple" },
];

export default function BottomBar() {
  const { thumbnailSize, setThumbnailSize, gridColumns } = useUIStore();


  const fewer = () => {
    if (gridColumns <= 1) return;
    const approxWidth = gridColumns * thumbnailSize;
    setThumbnailSize(Math.ceil(approxWidth / (gridColumns - 1)));
  };

  const more = () => {
    if (gridColumns <= 1) return;
    const approxWidth = gridColumns * thumbnailSize;
    setThumbnailSize(Math.floor(approxWidth / (gridColumns + 1)));
  };

  return (
    <div className="bottombar">
      {/* Left: rating + color labels */}
      <div className="flex items-center">
        <BauhausTooltip content="reject" placement="top">
          <button className="bottombar-btn">
            <Ban size={12} />
          </button>
        </BauhausTooltip>

        {[1, 2, 3, 4, 5].map((n) => (
          <BauhausTooltip key={n} content={`rate ${n} stars`} placement="top">
            <button className="bottombar-star">
              <Star size={12} />
            </button>
          </BauhausTooltip>
        ))}

        <div className="bottombar-spacer" />

        {COLORS.map((c) => (
          <BauhausTooltip key={c.key} content={`color label: ${c.key}`} placement="top">
            <button className="bottombar-color">
              <div
                className="bottombar-color-dot"
                style={{ backgroundColor: `var(${c.var})` }}
              />
            </button>
          </BauhausTooltip>
        ))}

        <BauhausTooltip content="remove color label" placement="top">
          <button className="bottombar-btn">
            <CircleOff size={12} />
          </button>
        </BauhausTooltip>
      </div>

      {/* Center: thumbs per row */}
      <div className="bottombar-pill">
        <span className="bottombar-pill-label">{gridColumns}</span>
        <BauhausTooltip content="fewer thumbnails per row" placement="top">
          <BauhausButton transparent icon={<Minus size={10} />} onClick={fewer} />
        </BauhausTooltip>
        <BauhausTooltip content="more thumbnails per row" placement="top">
          <BauhausButton transparent icon={<Plus size={10} />} onClick={more} />
        </BauhausTooltip>
      </div>
    </div>
  );
}
