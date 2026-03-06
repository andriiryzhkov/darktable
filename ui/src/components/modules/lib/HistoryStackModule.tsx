import CollapsibleModule from "../CollapsibleModule";

export default function HistoryStackModule() {
  return (
    <CollapsibleModule title="history stack" description="perform actions on the history stacks (edit histories) of the currently selected images">
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        copy / paste / compress history stack
      </p>
    </CollapsibleModule>
  );
}
