interface BauhausTabBarProps<T extends string> {
  tabs: readonly T[];
  value: T;
  justify?: "uniform" | "fit";
  onChange: (tab: T) => void;
}

export default function BauhausTabBar<T extends string>({
  tabs,
  value,
  justify = "fit",
  onChange,
}: BauhausTabBarProps<T>) {
  return (
    <div className="bauhaus-tab-bar" data-justify={justify}>
      {tabs.map((t) => (
        <button
          key={t}
          className={`bauhaus-tab ${value === t ? "bauhaus-tab--active" : ""}`}
          onClick={() => onChange(t)}
        >
          {t}
        </button>
      ))}
    </div>
  );
}
