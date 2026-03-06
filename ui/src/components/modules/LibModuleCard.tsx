import { type ReactNode } from "react";
import { CircleChevronRight, CircleChevronDown } from "lucide-react";
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
        open
          ? <CircleChevronDown size={12} className="module-chevron" />
          : <CircleChevronRight size={12} className="module-chevron" />
      }
    >
      {children}
    </ModuleCard>
  );
}
