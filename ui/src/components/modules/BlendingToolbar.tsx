import { useCallback, useMemo, useRef } from "react";
import {
  X, Circle, PenTool, SlidersHorizontal,
  Brush, SplinePointer, ArrowDownRight,
  SquareSquare, Menu, CircleOff, Eye, CirclePlus, CircleMinus,
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

/** Feathering guide enum values (from blend.h) */
const FEATHERING_GUIDE = {
  IN_BEFORE_BLUR: 0x01,
  OUT_BEFORE_BLUR: 0x02,
  IN_AFTER_BLUR: 0x05,
  OUT_AFTER_BLUR: 0x06,
} as const;

const FEATHERING_GUIDE_OPTIONS = [
  { value: FEATHERING_GUIDE.OUT_BEFORE_BLUR, label: "output before blur" },
  { value: FEATHERING_GUIDE.IN_BEFORE_BLUR, label: "input before blur" },
  { value: FEATHERING_GUIDE.OUT_AFTER_BLUR, label: "output after blur" },
  { value: FEATHERING_GUIDE.IN_AFTER_BLUR, label: "input after blur" },
];

/** DEVELOP_COMBINE_MASKS_POS flag for polarity toggle */
const COMBINE_MASKS_POS = 0x04;

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
  const maskUsage = useDevelopStore((s) => s.maskUsage);
  const modules = useDevelopStore((s) => s.modules);
  const showMasks = useDevelopStore((s) => s.showMasks);
  const toggleMasks = useDevelopStore((s) => s.toggleMasks);
  const assignMask = useDevelopStore((s) => s.assignMask);
  const clearModuleMasks = useDevelopStore((s) => s.clearModuleMasks);

  const blend = moduleInfo.blend;
  if (!blend) return null;

  const maskMode = blend.mask_mode ?? MASK_MODE.DISABLED;
  const isActive = maskMode !== MASK_MODE.DISABLED;
  const hasDrawn = (maskMode & MASK_MODE.DRAWN) !== 0;
  const hasMasks = !(moduleInfo.flags & IOP_FLAGS.NO_MASKS);
  const blendReversed = !!(blend.blend_mode & 0x80000000);

  // Count shapes used by this module's mask group
  const shapeCount = useMemo(() => {
    if (!blend.mask_id) return 0;
    const group = maskForms.find((f) => f.formid === blend.mask_id);
    return group?.children?.length ?? 0;
  }, [blend.mask_id, maskForms]);

  const maskLabel = shapeCount > 0
    ? `${shapeCount} shape${shapeCount !== 1 ? "s" : ""} used`
    : "no mask used";

  const polarityActive = !!(blend.mask_combine & COMBINE_MASKS_POS);

  // Build drawn mask combo: current shapes belonging to this module's group
  const myGroupChildIds = useMemo(() => {
    if (!blend.mask_id) return new Set<number>();
    const group = maskForms.find((f) => f.formid === blend.mask_id);
    return new Set((group?.children ?? []).map((c) => c.formid));
  }, [blend.mask_id, maskForms]);

  // Available shapes not already in this module's group (non-group forms)
  const availableShapes = useMemo(() => {
    return maskForms.filter(
      (f) => f.type_name !== "group" && !myGroupChildIds.has(f.formid),
    );
  }, [maskForms, myGroupChildIds]);

  // Other modules that have mask groups (for "use same shapes as")
  const otherModuleGroups = useMemo(() => {
    return maskUsage.filter(
      (u) => !(u.op === op && u.instance === instance) && u.mask_id,
    );
  }, [maskUsage, op, instance]);

  // Build combo groups and options
  const drawnMaskComboGroups = useMemo(() => {
    const groups: { label: string; options: string[] }[] = [];
    if (shapeCount > 0) {
      groups.push({ label: "", options: ["no mask used"] });
    }
    if (availableShapes.length > 0) {
      groups.push({
        label: "add existing shape",
        options: availableShapes.map((f) => `shape:${f.formid}:${f.name}`),
      });
    }
    if (otherModuleGroups.length > 0) {
      groups.push({
        label: "use same shapes as",
        options: otherModuleGroups.map((u) => {
          const mod = modules.find((m) => m.op === u.op && m.instance === u.instance);
          const label = mod?.name ?? u.module_name ?? u.op;
          return `module:${u.op}:${u.instance}:${label}`;
        }),
      });
    }
    return groups;
  }, [shapeCount, availableShapes, otherModuleGroups, modules]);

  const handleDrawnMaskComboChange = useCallback(
    (selected: string) => {
      if (selected === "no mask used") {
        clearModuleMasks(op, instance);
      } else if (selected.startsWith("shape:")) {
        const formid = parseInt(selected.split(":")[1], 10);
        if (!isNaN(formid)) assignMask(formid, op, instance);
      } else if (selected.startsWith("module:")) {
        // Copy shapes from another module's group
        const parts = selected.split(":");
        const srcOp = parts[1];
        const srcInstance = parseInt(parts[2], 10);
        const srcUsage = maskUsage.find((u) => u.op === srcOp && u.instance === srcInstance);
        if (srcUsage) {
          const srcGroup = maskForms.find((f) => f.formid === srcUsage.mask_id);
          if (srcGroup?.children) {
            // Assign each shape from the source group
            for (const child of srcGroup.children) {
              assignMask(child.formid, op, instance);
            }
          }
        }
      }
    },
    [op, instance, assignMask, clearModuleMasks, maskUsage, maskForms],
  );

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

  const handleTogglePolarity = useCallback(() => {
    const newCombine = blend.mask_combine ^ COMBINE_MASKS_POS;
    setBlendParam(op, instance, "mask_combine", newCombine);
  }, [op, instance, blend.mask_combine, setBlendParam]);

  // --- Throttled slider helpers ---
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

  // Generic throttled blend param handler for refinement sliders
  const makeSliderHandlers = useCallback(
    (param: string, scale = 1) => {
      let busy = false;
      let pending: number | null = null;
      const onChange = async (value: number) => {
        pending = value;
        if (busy) return;
        busy = true;
        try {
          while (pending !== null) {
            const v = pending;
            pending = null;
            await setBlendParam(op, instance, param, v * scale, true);
          }
        } finally {
          busy = false;
        }
      };
      const onRelease = (value: number) => {
        setBlendParam(op, instance, param, value * scale);
      };
      return { onChange, onRelease };
    },
    [op, instance, setBlendParam],
  );

  // Memoize slider handlers for each refinement param
  const detailsHandlers = useMemo(() => makeSliderHandlers("details"), [makeSliderHandlers]);
  const featheringRadiusHandlers = useMemo(() => makeSliderHandlers("feathering_radius"), [makeSliderHandlers]);
  const blurRadiusHandlers = useMemo(() => makeSliderHandlers("blur_radius"), [makeSliderHandlers]);
  const brightnessHandlers = useMemo(() => makeSliderHandlers("brightness"), [makeSliderHandlers]);
  const contrastHandlers = useMemo(() => makeSliderHandlers("contrast"), [makeSliderHandlers]);

  const handleFeatheringGuideChange = useCallback(
    (label: string) => {
      const opt = FEATHERING_GUIDE_OPTIONS.find((o) => o.label === label);
      if (opt) setBlendParam(op, instance, "feathering_guide", opt.value);
    },
    [op, instance, setBlendParam],
  );

  const currentFeatheringLabel = FEATHERING_GUIDE_OPTIONS.find(
    (o) => o.value === blend.feathering_guide,
  )?.label ?? "output before blur";

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

      {/* Drawn mask section */}
      {isActive && hasMasks && hasDrawn && (
        <BauhausSection
          header={
            <BauhausCombo
              label="drawn mask"
              value={maskLabel}
              groups={drawnMaskComboGroups}
              onChange={handleDrawnMaskComboChange}
              formatOption={(opt) => {
                if (opt.startsWith("shape:")) return opt.split(":").slice(2).join(":");
                if (opt.startsWith("module:")) return opt.split(":").slice(3).join(":");
                return opt;
              }}
              actionIcon={polarityActive ? <CircleMinus size={14} /> : <CirclePlus size={14} />}
              onAction={handleTogglePolarity}
            />
          }
        >
          <div className="blending-shapes blending-shapes-right">
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
            <BauhausTooltip content="show and edit mask elements" placement="bottom">
              <BauhausButton
                icon={<Eye size={12} />}
                transparent
                active={showMasks}
                onClick={toggleMasks}
              />
            </BauhausTooltip>
          </div>
        </BauhausSection>
      )}

      {/* Mask refinement section */}
      {isActive && (hasDrawn || maskMode === MASK_MODE.PARAMETRIC || maskMode === MASK_MODE.RASTER) && (
        <BauhausSection title="mask refinement">
          <BauhausSlider
            label="details threshold"
            value={blend.details}
            min={-1}
            max={1}
            step={0.01}
            defaultValue={0}
            format={(v) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(0)}%`}
            onChange={detailsHandlers.onChange}
            onRelease={detailsHandlers.onRelease}
          />
          <BauhausCombo
            label="feathering guide"
            value={currentFeatheringLabel}
            options={FEATHERING_GUIDE_OPTIONS.map((o) => o.label)}
            onChange={handleFeatheringGuideChange}
          />
          <BauhausSlider
            label="feathering radius"
            value={blend.feathering_radius}
            min={0}
            max={250}
            step={0.1}
            defaultValue={0}
            format={(v) => `${v.toFixed(1)} px`}
            onChange={featheringRadiusHandlers.onChange}
            onRelease={featheringRadiusHandlers.onRelease}
          />
          <BauhausSlider
            label="blurring radius"
            value={blend.blur_radius}
            min={0}
            max={100}
            step={0.1}
            defaultValue={0}
            format={(v) => `${v.toFixed(1)} px`}
            onChange={blurRadiusHandlers.onChange}
            onRelease={blurRadiusHandlers.onRelease}
          />
          <BauhausSlider
            label="mask opacity"
            value={blend.brightness}
            min={-1}
            max={1}
            step={0.01}
            defaultValue={0}
            format={(v) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(0)}%`}
            onChange={brightnessHandlers.onChange}
            onRelease={brightnessHandlers.onRelease}
          />
          <BauhausSlider
            label="mask contrast"
            value={blend.contrast}
            min={-1}
            max={1}
            step={0.01}
            defaultValue={0}
            format={(v) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(0)}%`}
            onChange={contrastHandlers.onChange}
            onRelease={contrastHandlers.onRelease}
          />
        </BauhausSection>
      )}
    </div>
  );
}
