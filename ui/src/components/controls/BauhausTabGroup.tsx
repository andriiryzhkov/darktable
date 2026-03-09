import { useState, type ReactNode } from "react";
import BauhausTabBar from "./BauhausTabBar";

interface BauhausTabGroupProps<T extends string> {
  tabs: readonly T[];
  defaultTab?: T;
  tab?: T;
  justify?: "uniform" | "fit";
  onTabChange?: (tab: T) => void;
  children: (tab: T) => ReactNode;
}

export default function BauhausTabGroup<T extends string>({
  tabs,
  defaultTab,
  tab: controlledTab,
  justify,
  onTabChange,
  children,
}: BauhausTabGroupProps<T>) {
  const [internalTab, setInternalTab] = useState<T>(defaultTab ?? tabs[0]);
  const activeTab = controlledTab ?? internalTab;

  const handleChange = (t: T) => {
    setInternalTab(t);
    onTabChange?.(t);
  };

  return (
    <>
      <BauhausTabBar tabs={tabs} value={activeTab} justify={justify} onChange={handleChange} />
      <div className="bauhaus-tab-stack">
        {tabs.map((t) => (
          <div
            key={t}
            className={`bauhaus-tab-panel${t !== activeTab ? " bauhaus-tab-panel--hidden" : ""}`}
          >
            {children(t)}
          </div>
        ))}
      </div>
    </>
  );
}
