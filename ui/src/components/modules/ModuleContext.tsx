import { createContext, useContext } from "react";

interface ModuleContextValue {
  op: string;
  view: string;
}

const ModuleContext = createContext<ModuleContextValue | null>(null);

export const ModuleProvider = ModuleContext.Provider;

export function useModuleContext(): ModuleContextValue {
  const ctx = useContext(ModuleContext);
  if (!ctx) throw new Error("useModuleContext must be used within ModuleProvider");
  return ctx;
}
