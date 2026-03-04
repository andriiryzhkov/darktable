import { Suspense } from "react";
import {
  getLibModules,
  VIEW_LIGHTTABLE,
  PANEL_RIGHT_CENTER,
} from "../modules/registry";

const modules = getLibModules(VIEW_LIGHTTABLE, PANEL_RIGHT_CENTER);

export default function RightSidebarModules() {
  return (
    <Suspense fallback={null}>
      {modules.map((m) => (
        <m.component key={m.op} />
      ))}
    </Suspense>
  );
}
