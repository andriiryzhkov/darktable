import CollapsibleModule from "../CollapsibleModule";

export default function HistoryStackModule() {
  return (
    <CollapsibleModule title="history stack">
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        copy / paste / compress history stack
      </p>
    </CollapsibleModule>
  );
}
