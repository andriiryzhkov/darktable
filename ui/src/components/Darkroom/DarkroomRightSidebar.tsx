import { useState, useMemo } from "react";
import { Search } from "lucide-react";
import {
  DARKROOM_MODULE_GROUPS,
  DARKROOM_MODULES,
  type ModuleGroup,
} from "./darkroomModules";
import DarkroomModuleCard from "./DarkroomModuleCard";

export default function DarkroomRightSidebar() {
  const [activeGroup, setActiveGroup] = useState<ModuleGroup>("active");
  const [searchQuery, setSearchQuery] = useState("");

  const filteredModules = useMemo(() => {
    let modules = DARKROOM_MODULES;

    if (activeGroup === "active") {
      modules = modules.filter((m) => m.enabled);
    } else if (activeGroup !== "favorites") {
      modules = modules.filter((m) => m.group === activeGroup);
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
        {filteredModules.map((mod) => (
          <DarkroomModuleCard key={mod.op} module={mod} />
        ))}
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
