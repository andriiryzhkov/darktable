import { useState, useCallback, Suspense } from "react";
import { RotateCcw, Menu, Copy, Power, CircleDot, AlertTriangle } from "lucide-react";
import type { IopModuleDef } from "./registry";
import BauhausButton from "../controls/BauhausButton";
import { useDevelopStore, getEnabledOps } from "../../stores/developStore";

/** Detect pipeline trouble messages for known modules */
function useModuleTrouble(op: string): string | null {
  const historyItems = useDevelopStore((s) => s.historyItems);
  const temperatureParams = useDevelopStore((s) => s.temperatureParams);
  const enabledOps = getEnabledOps(historyItems);

  if (op === "temperature") {
    const tempEnabled = enabledOps.has("temperature");
    const colorCalEnabled = enabledOps.has("channelmixerrgb");
    const preset = temperatureParams?.preset;
    const isD65 = preset === 3 || preset === 4;
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
  const trouble = useModuleTrouble(module.op);

  const enabled = getEnabledOps(historyItems).has(module.op);
  const isMandatory = historyItems.some((h) => h.op === module.op && h.mandatory);

  const toggleEnabled = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (isMandatory) return;
      enableModule(module.op, !enabled);
    },
    [module.op, enabled, enableModule, isMandatory],
  );

  const Component = module.component;

  return (
    <div className="module-wrapper" data-open={open}>
      <div className="module-header" onClick={() => setOpen(!open)}>
        <span
          className="darkroom-module-toggle"
          data-enabled={enabled}
          onClick={toggleEnabled}
        >
          {isMandatory ? <CircleDot size={12} /> : <Power size={12} />}
        </span>

        <span className="flex-1">{module.name}</span>
        {trouble && (
          <span className="module-trouble-icon" title={trouble}>
            <AlertTriangle size={12} />
          </span>
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
