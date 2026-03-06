import { type ReactNode } from "react";
import { CircleChevronRight, CircleChevronDown } from "lucide-react";
import BauhausButton from "../controls/BauhausButton";
import BauhausTooltip from "../controls/BauhausTooltip";
import ModuleCard from "./ModuleCard";

interface LibModuleCardProps {
  title: string;
  description?: string;
  defaultOpen?: boolean;
  onReset?: () => void;
  extraButtons?: ReactNode;
  children: ReactNode;
}

export default function LibModuleCard({
  title,
  description,
  defaultOpen = false,
  onReset,
  extraButtons,
  children,
}: LibModuleCardProps) {
  return (
    <ModuleCard
      title={title}
      tooltip={description ? <div className="module-desc-tooltip">{description}</div> : undefined}
      defaultOpen={defaultOpen}
      onReset={onReset}
      extraButtons={extraButtons}
      leftIcon={(open) =>
        <BauhausTooltip content="show module" placement="bottom">
          <span className="module-power">
            <BauhausButton
              icon={open
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
