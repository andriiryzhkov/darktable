import LibModuleCard from "../LibModuleCard";

export default function HistoryStackModule() {
  return (
    <LibModuleCard title="history stack" description="perform actions on the history stacks (edit histories) of the currently selected images">
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        copy / paste / compress history stack
      </p>
    </LibModuleCard>
  );
}
