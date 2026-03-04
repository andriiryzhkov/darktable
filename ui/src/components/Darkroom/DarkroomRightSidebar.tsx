import { useState, useMemo, Suspense } from "react";
import { Search } from "lucide-react";
import {
  IOP_MODULES,
  IOP_GROUP_BASIC,
  IOP_GROUP_TONE,
  IOP_GROUP_COLOR,
  IOP_GROUP_CORRECT,
  IOP_GROUP_EFFECT,
  type IopModuleDef,
} from "../modules/registry";
import {
  DARKROOM_MODULE_GROUPS,
  type ModuleGroup,
} from "./darkroomModules";
import ProcessingModuleCard from "../modules/ProcessingModuleCard";

// Map UI group id to IOP_GROUP bitmask
const GROUP_MAP: Record<string, number> = {
  basic: IOP_GROUP_BASIC,
  tone: IOP_GROUP_TONE,
  color: IOP_GROUP_COLOR,
  correct: IOP_GROUP_CORRECT,
  effect: IOP_GROUP_EFFECT,
};

export default function DarkroomRightSidebar() {
  const [activeGroup, setActiveGroup] = useState<ModuleGroup>("active");
  const [searchQuery, setSearchQuery] = useState("");

  const filteredModules = useMemo(() => {
    let modules: IopModuleDef[] = IOP_MODULES;

    if (activeGroup === "active") {
      // TODO: filter by actually enabled modules from develop store
      modules = [...modules];
    } else if (activeGroup !== "favorites") {
      const mask = GROUP_MAP[activeGroup];
      if (mask) {
        modules = modules.filter((m) => (m.defaultGroup & mask) !== 0);
      }
    }

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      modules = modules.filter(
        (m) =>
          m.name.toLowerCase().includes(q) ||
          m.tags?.some((t) => t.toLowerCase().includes(q)),
      );
    }

    return modules;
  }, [activeGroup, searchQuery]);

  return (
    <div className="flex flex-col h-full">
      {/* Module group tabs */}
      <div className="darkroom-group-tabs">
        {DARKROOM_MODULE_GROUPS.map((group) => (
          <button
            key={group.id}
            className="darkroom-group-tab"
            data-active={activeGroup === group.id}
            title={group.label}
            onClick={() => setActiveGroup(group.id)}
          >
            {group.icon}
          </button>
        ))}
      </div>

      {/* Search bar */}
      <div className="darkroom-search">
        <Search size={12} className="darkroom-search-icon" />
        <input
          type="text"
          placeholder="search modules by name or tag"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="darkroom-search-input"
        />
      </div>

      {/* Module list */}
      <div className="flex-1 overflow-y-auto">
        <Suspense fallback={null}>
          {filteredModules.map((mod) => (
            <ProcessingModuleCard
              key={mod.op}
              module={mod}
              enabled={true}
            />
          ))}
        </Suspense>
        {filteredModules.length === 0 && (
          <p
            className="text-xs text-center py-4"
            style={{ color: "var(--disabled-fg-color)" }}
          >
            no modules found
          </p>
        )}
      </div>
    </div>
  );
}
