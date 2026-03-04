import { Suspense } from "react";
import {
  getLibModules,
  VIEW_LIGHTTABLE,
  PANEL_LEFT_CENTER,
} from "../modules/registry";

const modules = getLibModules(VIEW_LIGHTTABLE, PANEL_LEFT_CENTER);

export default function LeftSidebarModules() {
  return (
    <Suspense fallback={null}>
      {modules.map((m) => (
        <m.component key={m.op} />
      ))}
    </Suspense>
  );
}
