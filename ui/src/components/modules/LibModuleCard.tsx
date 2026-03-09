import { type ReactNode } from "react";
import ModuleCard from "./ModuleCard";
import { useModuleContext } from "./ModuleContext";
import { useModuleExpanded } from "../../hooks/useModuleExpanded";

interface LibModuleCardProps {
  title: string;
  description?: string;
  onReset?: () => void;
  rightButtons?: ReactNode;
  children: ReactNode;
}

export default function LibModuleCard({
  title,
  description,
  onReset,
  rightButtons,
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
      rightButtons={rightButtons}
      leftButton={{ kind: "chevron" }}
    >
      {children}
    </ModuleCard>
  );
}
