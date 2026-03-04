import { Suspense } from "react";
import {
  getLibModules,
  VIEW_DARKROOM,
  PANEL_LEFT_TOP,
  PANEL_LEFT_CENTER,
} from "../modules/registry";

const topModules = getLibModules(VIEW_DARKROOM, PANEL_LEFT_TOP);
const centerModules = getLibModules(VIEW_DARKROOM, PANEL_LEFT_CENTER);

export default function DarkroomLeftSidebar() {
  return (
    <Suspense fallback={null}>
      {topModules.map((m) => (
        <m.component key={m.op} />
      ))}
      {centerModules.map((m) => (
        <m.component key={m.op} />
      ))}
    </Suspense>
  );
}
