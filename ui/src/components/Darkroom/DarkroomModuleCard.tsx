import { useState, useCallback, useRef } from "react";
import { RotateCcw, Menu, Copy, Power } from "lucide-react";
import type { DarkroomModuleDef } from "./darkroomModules";
import { useDevelopStore } from "../../stores/developStore";
import ModuleSlider from "../Sidebar/controls/ModuleSlider";

interface Props {
  module: DarkroomModuleDef;
}

export default function DarkroomModuleCard({ module }: Props) {
  const [open, setOpen] = useState(module.defaultOpen ?? false);
  const [enabled, setEnabled] = useState(module.enabled);

  const toggleEnabled = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      setEnabled((prev) => !prev);
    },
    [],
  );

  return (
    <div className="module-wrapper" data-open={open}>
      <button className="module-header" onClick={() => setOpen(!open)}>
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
          <span title="Multi-instance" className="module-action-btn">
            <Copy size={12} />
          </span>
          <span title="Reset" className="module-action-btn">
            <RotateCcw size={12} />
          </span>
          <span title="Presets" className="module-action-btn">
            <Menu size={12} />
          </span>
        </span>
      </button>

      {open && (
        <div className="module-content">
          <ModuleControls op={module.op} />
        </div>
      )}
    </div>
  );
}

function ModuleControls({ op }: { op: string }) {
  switch (op) {
    case "exposure":
      return <ExposureControls />;
    default:
      return (
        <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
          module controls not yet implemented
        </p>
      );
  }
}

function ExposureControls() {
  const { setExposure, setBlack } = useDevelopStore();
  const [exposureValue, setExposureValue] = useState(0);
  const [blackValue, setBlackValue] = useState(0);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleExposure = useCallback(
    (v: number) => {
      setExposureValue(v);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => setExposure(v), 150);
    },
    [setExposure],
  );

  const handleBlack = useCallback(
    (v: number) => {
      setBlackValue(v);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => setBlack(v), 150);
    },
    [setBlack],
  );

  return (
    <>
      <ModuleSlider
        label="exposure"
        value={exposureValue}
        min={-4}
        max={4}
        step={0.01}
        defaultValue={0}
        origin={0}
        format={(v) => `${v.toFixed(2)} EV`}
        onChange={handleExposure}
      />
      <ModuleSlider
        label="black level correction"
        value={blackValue}
        min={-0.1}
        max={0.1}
        step={0.001}
        defaultValue={0}
        origin={0}
        format={(v) => v.toFixed(4)}
        onChange={handleBlack}
      />
    </>
  );
}
