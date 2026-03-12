import { useCallback, useMemo, useRef } from "react";
import {
  X, Circle, PenTool, SlidersHorizontal,
  Brush, SplinePointer, ArrowDownRight,
  SquareSquare, Menu, CircleOff,
} from "lucide-react";
import BauhausButton from "../controls/BauhausButton";
import BauhausTooltip from "../controls/BauhausTooltip";
import BauhausSlider from "../controls/BauhausSlider";
import BauhausCombo from "../controls/BauhausCombo";
import BauhausSection from "../controls/BauhausSection";
import { useDevelopStore } from "../../stores/developStore";
import { MASK_MODE, BLEND_MODE, IOP_FLAGS } from "../../types/protocol";
import type { ModuleInfo } from "../../types/protocol";

const ICON_SIZE = 14;

/** Combined drawn+parametric icon: pen overlapping sliders */
function DrawnParametricIcon() {
  return (
    <span className="blending-combined-icon">
      <PenTool size={12} />
      <SlidersHorizontal size={10} />
    </span>
  );
}

const BLEND_MODE_OPTIONS = [
  // normal & arithmetic
  { value: BLEND_MODE.NORMAL2, label: "normal", group: "normal & arithmetic" },
  { value: BLEND_MODE.AVERAGE, label: "average", group: "normal & arithmetic" },
  { value: BLEND_MODE.DIFFERENCE2, label: "difference", group: "normal & arithmetic" },
  { value: BLEND_MODE.MULTIPLY, label: "multiply", group: "normal & arithmetic" },
  { value: BLEND_MODE.DIVIDE, label: "divide", group: "normal & arithmetic" },
  { value: BLEND_MODE.ADD, label: "addition", group: "normal & arithmetic" },
  { value: BLEND_MODE.SUBTRACT, label: "subtract", group: "normal & arithmetic" },
  { value: BLEND_MODE.GEOMETRIC_MEAN, label: "geometric mean", group: "normal & arithmetic" },
  { value: BLEND_MODE.HARMONIC_MEAN, label: "harmonic mean", group: "normal & arithmetic" },
  // color channel
  { value: BLEND_MODE.RGB_R, label: "RGB red channel", group: "color channel" },
  { value: BLEND_MODE.RGB_G, label: "RGB green channel", group: "color channel" },
  { value: BLEND_MODE.RGB_B, label: "RGB blue channel", group: "color channel" },
  // chromaticity & lightness
  { value: BLEND_MODE.LIGHTNESS, label: "lightness", group: "chromaticity & lightness" },
  { value: BLEND_MODE.CHROMATICITY, label: "chromaticity", group: "chromaticity & lightness" },
];

interface Props {
  op: string;
  instance: number;
  moduleInfo: ModuleInfo;
}

export default function BlendingToolbar({ op, instance, moduleInfo }: Props) {
  const setBlendParam = useDevelopStore((s) => s.setBlendParam);
  const createMask = useDevelopStore((s) => s.createMask);
  const startCreation = useDevelopStore((s) => s.startCreation);
  const creationTool = useDevelopStore((s) => s.creationTool);
  const creatingMaskId = useDevelopStore((s) => s.creatingMaskId);
  const maskForms = useDevelopStore((s) => s.maskForms);

  const blend = moduleInfo.blend;
  if (!blend) return null;

  const maskMode = blend.mask_mode ?? MASK_MODE.DISABLED;
  const isActive = maskMode !== MASK_MODE.DISABLED;
  const hasMasks = !(moduleInfo.flags & IOP_FLAGS.NO_MASKS);
  const blendReversed = !!(blend.blend_mode & 0x80000000);

  const setMaskMode = useCallback(
    (mode: number) => { setBlendParam(op, instance, "mask_mode", mode); },
    [op, instance, setBlendParam],
  );

  const handleBlendModeChange = useCallback(
    (label: string) => {
      const opt = BLEND_MODE_OPTIONS.find((o) => o.label === label);
      if (opt) setBlendParam(op, instance, "blend_mode", opt.value);
    },
    [op, instance, setBlendParam],
  );

  const handleToggleReverse = useCallback((pressed: boolean) => {
    const base = blend.blend_mode & 0xFF;
    const newMode = pressed ? ((base | 0x80000000) >>> 0) : base;
    setBlendParam(op, instance, "blend_mode", newMode);
  }, [op, instance, blend.blend_mode, setBlendParam]);

  const busyRef = useRef(false);
  const pendingRef = useRef<number | null>(null);

  const handleOpacityChange = useCallback(
    async (value: number) => {
      pendingRef.current = value;
      if (busyRef.current) return;
      busyRef.current = true;
      try {
        while (pendingRef.current !== null) {
          const v = pendingRef.current;
          pendingRef.current = null;
          await setBlendParam(op, instance, "opacity", v * 100, true);
        }
      } finally {
        busyRef.current = false;
      }
    },
    [op, instance, setBlendParam],
  );

  const handleOpacityRelease = useCallback(
    (value: number) => { setBlendParam(op, instance, "opacity", value * 100); },
    [op, instance, setBlendParam],
  );


  const currentBlendLabel = BLEND_MODE_OPTIONS.find(
    (o) => o.value === (blend.blend_mode & 0xFF),
  )?.label ?? "normal";

  const blendGroups = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const o of BLEND_MODE_OPTIONS) {
      const list = map.get(o.group);
      if (list) list.push(o.label);
      else map.set(o.group, [o.label]);
    }
    return Array.from(map, ([label, options]) => ({ label, options }));
  }, []);

  return (
    <div className="blending-toolbar">
      {/* Mask mode buttons */}
      <div className="blending-mask-modes">
        <BauhausTooltip content="off" placement="bottom">
          <BauhausButton
            icon={<X size={ICON_SIZE} />}
            transparent
            onClick={() => setMaskMode(MASK_MODE.DISABLED)}
          />
        </BauhausTooltip>
        <BauhausTooltip content="uniformly" placement="bottom">
          <BauhausButton
            icon={<Circle size={ICON_SIZE} />}
            transparent
            active={maskMode === MASK_MODE.ENABLED}
            onClick={() => setMaskMode(MASK_MODE.ENABLED)}
          />
        </BauhausTooltip>
        {hasMasks && (
          <BauhausTooltip content="drawn mask" placement="bottom">
            <BauhausButton
              icon={<PenTool size={ICON_SIZE} />}
              transparent
              active={maskMode === MASK_MODE.DRAWN}
              onClick={() => setMaskMode(MASK_MODE.DRAWN)}
            />
          </BauhausTooltip>
        )}
        <BauhausTooltip content="parametric mask" placement="bottom">
          <BauhausButton
            icon={<SlidersHorizontal size={ICON_SIZE} />}
            transparent
            active={maskMode === MASK_MODE.PARAMETRIC}
            onClick={() => setMaskMode(MASK_MODE.PARAMETRIC)}
          />
        </BauhausTooltip>
        {hasMasks && (
          <BauhausTooltip content="drawn & parametric" placement="bottom">
            <BauhausButton
              icon={<DrawnParametricIcon />}
              transparent
              active={maskMode === MASK_MODE.DRAWN_PARAMETRIC}
              onClick={() => setMaskMode(MASK_MODE.DRAWN_PARAMETRIC)}
            />
          </BauhausTooltip>
        )}
        {hasMasks && (
          <BauhausTooltip content="raster mask" placement="bottom">
            <BauhausButton
              icon={<SquareSquare size={ICON_SIZE} />}
              transparent
              active={maskMode === MASK_MODE.RASTER}
              onClick={() => setMaskMode(MASK_MODE.RASTER)}
            />
          </BauhausTooltip>
        )}
        <BauhausTooltip content="blending options" placement="bottom">
          <BauhausButton
            icon={<Menu size={ICON_SIZE} />}
            transparent
          />
        </BauhausTooltip>
      </div>

      {/* Blend controls (shown when masking is active) */}
      {isActive && (
        <BauhausSection title="blend mask">
          <div className="blending-mode-row">
            <BauhausCombo
              label="mode"
              value={currentBlendLabel}
              groups={blendGroups}
              onChange={handleBlendModeChange}
            />
            <BauhausTooltip content="toggle blend order" placement="bottom">
              <BauhausButton
                icon={<CircleOff size={14} />}
                transparent
                toggle
                active={blendReversed}
                onToggle={handleToggleReverse}
              />
            </BauhausTooltip>
          </div>
          <BauhausSlider
            label="opacity"
            value={blend.opacity / 100}
            min={0}
            max={1}
            step={0.01}
            defaultValue={1}
            format={(v) => `${(v * 100).toFixed(0)}%`}
            onChange={handleOpacityChange}
            onRelease={handleOpacityRelease}
          />
        </BauhausSection>
      )}

      {/* Drawn mask shape tools (when drawn mask mode active) */}
      {isActive && hasMasks && (maskMode & MASK_MODE.DRAWN) !== 0 && (
        <div className="blending-shapes">
          <BauhausTooltip content="add circle" placement="bottom">
            <BauhausButton
              icon={<Circle size={12} />}
              transparent
              active={creatingMaskId !== null && maskForms.find((f) => f.formid === creatingMaskId)?.type_name === "circle"}
              onClick={() => createMask("circle", { center: [0.5, 0.5], radius: 0.05, border: 0.05, op, instance, _creation: true })}
            />
          </BauhausTooltip>
          <BauhausTooltip content="add ellipse" placement="bottom">
            <BauhausButton
              icon={<Circle size={12} style={{ transform: "scaleX(0.7)" }} />}
              transparent
              active={creatingMaskId !== null && maskForms.find((f) => f.formid === creatingMaskId)?.type_name === "ellipse"}
              onClick={() => createMask("ellipse", { center: [0.5, 0.5], radius: [0.05, 0.03535], border: 0.05, rotation: 90, op, instance, _creation: true })}
            />
          </BauhausTooltip>
          <BauhausTooltip content="add path" placement="bottom">
            <BauhausButton
              icon={<SplinePointer size={12} />}
              transparent
              active={creationTool === "path"}
              onClick={() => startCreation("path", op, instance)}
            />
          </BauhausTooltip>
          <BauhausTooltip content="add brush" placement="bottom">
            <BauhausButton
              icon={<Brush size={12} />}
              transparent
              active={creationTool === "brush"}
              onClick={() => startCreation("brush", op, instance)}
            />
          </BauhausTooltip>
          <BauhausTooltip content="add gradient" placement="bottom">
            <BauhausButton
              icon={<ArrowDownRight size={12} />}
              transparent
              active={creatingMaskId !== null && maskForms.find((f) => f.formid === creatingMaskId)?.type_name === "gradient"}
              onClick={() => createMask("gradient", { anchor: [0.5, 0.5], rotation: 0, compression: 0.05, steepness: 4, curvature: 0, state: 2, op, instance, _creation: true })}
            />
          </BauhausTooltip>
        </div>
      )}
    </div>
  );
}
