import { type ReactNode } from "react";
import { CircleChevronRight, CircleChevronDown } from "lucide-react";
import BauhausButton from "../controls/BauhausButton";
import BauhausTooltip from "../controls/BauhausTooltip";
import ModuleCard from "./ModuleCard";
import { useModuleContext } from "./ModuleContext";
import { useModuleExpanded } from "../../hooks/useModuleExpanded";

interface LibModuleCardProps {
  title: string;
  description?: string;
  onReset?: () => void;
  extraButtons?: ReactNode;
  children: ReactNode;
}

export default function LibModuleCard({
  title,
  description,
  onReset,
  extraButtons,
  children,
}: LibModuleCardProps) {
  const { op, view } = useModuleContext();
  const { open, setOpen } = useModuleExpanded(view, op);

  return (
    <ModuleCard
      title={title}
      tooltip={description ? <div className="module-desc-tooltip">{description}</div> : undefined}
      open={open}
      onToggle={setOpen}
      onReset={onReset}
      extraButtons={extraButtons}
      leftIcon={(isOpen) =>
        <BauhausTooltip content="show module" placement="bottom">
          <span className="module-power">
            <BauhausButton
              icon={isOpen
                ? <CircleChevronDown size={12} />
                : <CircleChevronRight size={12} />
              }
            />
          </span>
        </BauhausTooltip>
      }
    >
      {children}
    </ModuleCard>
  );
}
