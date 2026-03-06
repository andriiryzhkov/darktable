import { useCallback, useEffect, useRef, Suspense } from "react";
import { Copy, Power, CircleDot, AlertTriangle, Crosshair, ArrowRightToLine, Workflow, ArrowRightFromLine } from "lucide-react";
import type { IopModuleDef } from "./registry";
import type { ModuleDescription } from "../../types/protocol";
import BauhausButton from "../controls/BauhausButton";
import BauhausTooltip from "../controls/BauhausTooltip";
import ModuleCard from "./ModuleCard";
import { useDevelopStore, getEnabledOps } from "../../stores/developStore";
import { useModuleExpanded } from "../../hooks/useModuleExpanded";

/** Detect pipeline trouble messages for known modules */
function useModuleTrouble(op: string): string | null {
  const historyItems = useDevelopStore((s) => s.historyItems);
  const temperatureParams = useDevelopStore((s) => s.temperatureParams);
  const enabledOps = getEnabledOps(historyItems);

  if (op === "temperature") {
    const tempEnabled = enabledOps.has("temperature");
    const colorCalEnabled = enabledOps.has("channelmixerrgb");
    if (!temperatureParams) return null;
    const isD65 = temperatureParams.preset === 3 || temperatureParams.preset === 4;
    if (tempEnabled && colorCalEnabled && !isD65) return "white balance applied twice";
  }

  return null;
}

const ICON_SIZE = 10;

function ModuleDescriptionTooltip({ desc }: { desc: ModuleDescription }) {
  return (
    <div className="module-desc-tooltip">
      <div className="module-desc-main">{desc.main}</div>
      <div className="module-desc-grid">
        <Crosshair size={ICON_SIZE} strokeWidth={2.5} />
        <strong>purpose:</strong>
        <span>{desc.purpose}</span>

        <ArrowRightToLine size={ICON_SIZE} strokeWidth={2.5} />
        <strong>input:</strong>
        <span>{desc.input}</span>

        <Workflow size={ICON_SIZE} strokeWidth={2.5} />
        <strong>process:</strong>
        <span>{desc.process}</span>

        <ArrowRightFromLine size={ICON_SIZE} strokeWidth={2.5} />
        <strong>output:</strong>
        <span>{desc.output}</span>
      </div>
    </div>
  );
}

interface Props {
  module: IopModuleDef;
}

export default function IopModuleCard({ module }: Props) {
  const { open, setOpen } = useModuleExpanded("darkroom", module.op, false, "iop");
  const historyItems = useDevelopStore((s) => s.historyItems);
  const enableModule = useDevelopStore((s) => s.enableModule);
  const resetModule = useDevelopStore((s) => s.resetModule);
  const focusModuleOp = useDevelopStore((s) => s.focusModuleOp);
  const description = useDevelopStore((s) => s.moduleDescriptions[module.op]);
  const trouble = useModuleTrouble(module.op);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (focusModuleOp === module.op) {
      setOpen(true);
      useDevelopStore.setState({ focusModuleOp: null });
      requestAnimationFrame(() => {
        wrapperRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      });
    }
  }, [focusModuleOp, module.op]);

  const enabled = getEnabledOps(historyItems).has(module.op);
  const isMandatory = historyItems.some((h) => h.op === module.op && h.mandatory);

  const toggleEnabled = useCallback(() => {
    if (isMandatory) return;
    enableModule(module.op, !enabled);
  }, [module.op, enabled, enableModule, isMandatory]);

  const Component = module.component;

  return (
    <ModuleCard
      title={module.name}
      tooltip={description ? <ModuleDescriptionTooltip desc={description} /> : undefined}
      open={open}
      onToggle={setOpen}
      wrapperRef={wrapperRef}
      leftIcon={
        isMandatory ? (
          <BauhausTooltip content={`'${module.name}' is switched on`} placement="bottom">
            <CircleDot size={12} className="module-mandatory-icon" />
          </BauhausTooltip>
        ) : (
          <BauhausTooltip content={`'${module.name}' is switched ${enabled ? "on" : "off"}`} placement="bottom">
            <span className="module-power" onClick={(e) => e.stopPropagation()}>
              <BauhausButton
                icon={<Power size={12} />}
                active={enabled}
                onClick={toggleEnabled}
              />
            </span>
          </BauhausTooltip>
        )
      }
      afterTitle={
        trouble ? (
          <BauhausTooltip content={trouble} placement="left">
            <span className="module-trouble-icon">
              <AlertTriangle size={12} />
            </span>
          </BauhausTooltip>
        ) : undefined
      }
      onReset={() => resetModule(module.op)}
      resetTooltip={<span style={{ whiteSpace: "pre" }}>{"reset parameters\nctrl-click to reapply any automatic presets"}</span>}
      extraButtons={
        <BauhausTooltip content={<span style={{ whiteSpace: "pre" }}>{"multiple instance action\nright-click creates new instance"}</span>} placement="bottom">
          <BauhausButton icon={<Copy size={12} />} />
        </BauhausTooltip>
      }
      presetsTooltip={<span style={{ whiteSpace: "pre" }}>{"presets\nright-click to apply on new instance"}</span>}
    >
      <Suspense fallback={null}>
        <Component />
      </Suspense>
    </ModuleCard>
  );
}
