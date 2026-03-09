import { createContext, useContext, type ReactNode } from "react";

interface IopModuleContextValue {
  setIndicator: (indicator: ReactNode | null) => void;
}

const IopModuleContext = createContext<IopModuleContextValue | null>(null);

export const IopModuleProvider = IopModuleContext.Provider;

export function useIopModuleContext(): IopModuleContextValue {
  const ctx = useContext(IopModuleContext);
  if (!ctx) throw new Error("useIopModuleContext must be used within IopModuleProvider");
  return ctx;
}
