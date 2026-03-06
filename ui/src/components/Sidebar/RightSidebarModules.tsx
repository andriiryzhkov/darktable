import { Suspense } from "react";
import {
  getLibModules,
  VIEW_LIGHTTABLE,
  PANEL_RIGHT_CENTER,
} from "../modules/registry";
import { ModuleProvider } from "../modules/ModuleContext";

const modules = getLibModules(VIEW_LIGHTTABLE, PANEL_RIGHT_CENTER);

export default function RightSidebarModules() {
  return (
    <Suspense fallback={null}>
      {modules.map((m) => (
        <ModuleProvider key={m.op} value={{ op: m.op, view: "lighttable" }}>
          <m.component />
        </ModuleProvider>
      ))}
    </Suspense>
  );
}
