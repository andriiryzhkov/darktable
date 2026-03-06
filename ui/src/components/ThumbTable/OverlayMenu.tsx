import { useCallback } from "react";
import { Layers } from "lucide-react";
import { OverlayMode, ThumbTableMode } from "./types";
import { useOverlayStore } from "../../stores/overlayStore";
import BauhausDropdown from "../controls/BauhausDropdown";
import BauhausCheckbox from "../controls/BauhausCheckbox";
import BauhausInput from "../controls/BauhausInput";

const OVERLAY_LABELS: Record<OverlayMode, string> = {
  [OverlayMode.None]: "no overlays",
  [OverlayMode.HoverNormal]: "overlays on mouse hover",
  [OverlayMode.HoverExtended]: "extended overlays on mouse hover",
  [OverlayMode.AlwaysNormal]: "permanent overlays",
  [OverlayMode.AlwaysExtended]: "permanent extended overlays",
  [OverlayMode.Mixed]: "permanent overlays extended on mouse hover",
  [OverlayMode.HoverBlock]: "overlays block on mouse hover during (s)",
};

const OVERLAY_OPTIONS = Object.values(OVERLAY_LABELS);

const HOVER_BLOCK_LABEL = OVERLAY_LABELS[OverlayMode.HoverBlock];

const labelToMode = new Map(
  Object.entries(OVERLAY_LABELS).map(([k, v]) => [v, Number(k) as OverlayMode]),
);

interface Props {
  mode: ThumbTableMode;
}

export default function OverlayMenu({ mode }: Props) {
  const settings = useOverlayStore((s) => s.modes[mode]);
  const setOverlay = useOverlayStore((s) => s.setOverlay);
  const setTooltip = useOverlayStore((s) => s.setTooltip);
  const setBlockTimeout = useOverlayStore((s) => s.setBlockTimeout);

  const handleChange = useCallback(
    (label: string) => {
      const ov = labelToMode.get(label);
      if (ov !== undefined) setOverlay(mode, ov);
    },
    [mode, setOverlay],
  );

  const renderSuffix = useCallback(
    (opt: string) => {
      if (opt !== HOVER_BLOCK_LABEL) return null;
      return (
        <div className="bauhaus-dropdown-inline-input" onClick={(e) => e.stopPropagation()}>
          <BauhausInput
            type="integer"
            label=""
            value={settings.blockTimeout}
            min={-1}
            max={99}
            step={1}
            onChange={(v) => setBlockTimeout(mode, v)}
          />
        </div>
      );
    },
    [settings.blockTimeout, mode, setBlockTimeout],
  );

  return (
    <BauhausDropdown
      icon={<Layers size={14} />}
      options={OVERLAY_OPTIONS}
      value={OVERLAY_LABELS[settings.overlay]}
      onChange={handleChange}
      optionSuffix={renderSuffix}
      footer={
        <div className="bauhaus-dropdown-footer">
          <BauhausCheckbox
            label="show tooltip"
            checked={settings.tooltip}
            align="left"
            onChange={() => setTooltip(mode, !settings.tooltip)}
          />
        </div>
      }
    />
  );
}
