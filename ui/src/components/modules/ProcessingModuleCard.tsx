import { useState, useCallback, Suspense } from "react";
import { RotateCcw, Menu, Copy, Power } from "lucide-react";
import type { IopModuleDef } from "./registry";
import BauhausButton from "../controls/BauhausButton";

interface Props {
  module: IopModuleDef;
  enabled: boolean;
  defaultOpen?: boolean;
}

export default function ProcessingModuleCard({ module, enabled: initialEnabled, defaultOpen }: Props) {
  const [open, setOpen] = useState(defaultOpen ?? false);
  const [enabled, setEnabled] = useState(initialEnabled);

  const toggleEnabled = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      setEnabled((prev) => !prev);
    },
    [],
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
          <Power size={12} />
        </span>

        <span className="flex-1">{module.name}</span>

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
