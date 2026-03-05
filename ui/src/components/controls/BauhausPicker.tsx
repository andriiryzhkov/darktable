import { useCallback } from "react";
import { Pipette } from "lucide-react";
import { usePickerStore, type PickerMode } from "../../stores/pickerStore";

interface BauhausPickerProps {
  id: string;
  module: string;
  mode?: PickerMode;
}

export default function BauhausPicker({ id, module, mode = "area" }: BauhausPickerProps) {
  const active = usePickerStore((s) => s.active);
  const activate = usePickerStore((s) => s.activate);
  const isActive = active?.id === id;

  const handleClick = useCallback(() => {
    activate(id, module, mode);
  }, [id, module, mode, activate]);

  return (
    <div
      className="bauhaus-picker"
      data-active={isActive}
      onClick={handleClick}
    >
      <Pipette size={12} />
    </div>
  );
}
