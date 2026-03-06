import { useState, useCallback, useEffect, useRef, Suspense } from "react";
import { RotateCcw, Menu, Copy, Power, CircleDot, AlertTriangle } from "lucide-react";
import type { IopModuleDef } from "./registry";
import BauhausButton from "../controls/BauhausButton";
import BauhausTooltip from "../controls/BauhausTooltip";
import { useDevelopStore, getEnabledOps } from "../../stores/developStore";

/** Detect pipeline trouble messages for known modules */
function useModuleTrouble(op: string): string | null {
  const historyItems = useDevelopStore((s) => s.historyItems);
  const temperatureParams = useDevelopStore((s) => s.temperatureParams);
  const enabledOps = getEnabledOps(historyItems);

  if (op === "temperature") {
    const tempEnabled = enabledOps.has("temperature");
    const colorCalEnabled = enabledOps.has("channelmixerrgb");
    if (!temperatureParams) return null; // params not loaded yet
    const isD65 = temperatureParams.preset === 3 || temperatureParams.preset === 4;
    if (tempEnabled && colorCalEnabled && !isD65) return "white balance applied twice";
  }

  return null;
}

interface Props {
  module: IopModuleDef;
  defaultOpen?: boolean;
}

export default function ProcessingModuleCard({ module, defaultOpen }: Props) {
  const [open, setOpen] = useState(defaultOpen ?? false);
  const historyItems = useDevelopStore((s) => s.historyItems);
  const enableModule = useDevelopStore((s) => s.enableModule);
  const focusModuleOp = useDevelopStore((s) => s.focusModuleOp);
  const trouble = useModuleTrouble(module.op);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // React to focusModuleOp: open and scroll into view
  useEffect(() => {
    if (focusModuleOp === module.op) {
      setOpen(true);
      // Clear after consuming so it can be re-triggered
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
    <div ref={wrapperRef} className="module-wrapper" data-open={open}>
      <div className="module-header" onClick={() => setOpen(!open)}>
        {isMandatory ? (
          <CircleDot size={12} className="module-mandatory-icon" />
        ) : (
          <span className="module-power" onClick={(e) => e.stopPropagation()}>
            <BauhausButton
              icon={<Power size={12} />}
              active={enabled}
              onClick={toggleEnabled}
            />
          </span>
        )}

        <span className="flex-1">{module.name}</span>
        {trouble && (
          <BauhausTooltip content={trouble} placement="left">
            <span className="module-trouble-icon">
              <AlertTriangle size={12} />
            </span>
          </BauhausTooltip>
        )}

        <span
          className="module-actions"
          onClick={(e) => e.stopPropagation()}
        >
          <BauhausButton icon={<Copy size={12} />} />
          <BauhausButton icon={<RotateCcw size={12} />} />
          <BauhausButton icon={<Menu size={12} />} />
        </span>
      </div>

      {open && (
        <div className="module-content">
          <Suspense fallback={null}>
            <Component />
          </Suspense>
        </div>
      )}
    </div>
  );
}
