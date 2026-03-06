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
    <div className="flex flex-col flex-1 min-h-0">
      {/* Navigation — always visible, not scrollable */}
      <Suspense fallback={null}>
        {topModules.map((m) => (
          <m.component key={m.op} />
        ))}
      </Suspense>

      {/* Modules — scrollable, scrollbar on left via RTL trick */}
      <div className="flex-1 overflow-y-scroll min-h-0" style={{ direction: "rtl" }}>
        <div style={{ direction: "ltr" }}>
          <Suspense fallback={null}>
            {centerModules.map((m) => (
              <m.component key={m.op} />
            ))}
          </Suspense>
        </div>
      </div>
    </div>
  );
}
