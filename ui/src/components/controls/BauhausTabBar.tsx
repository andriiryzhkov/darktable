interface BauhausTabBarProps<T extends string> {
  tabs: readonly T[];
  value: T;
  onChange: (tab: T) => void;
}

export default function BauhausTabBar<T extends string>({
  tabs,
  value,
  onChange,
}: BauhausTabBarProps<T>) {
  return (
    <div className="bauhaus-tab-bar">
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
