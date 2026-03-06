import { useState, useMemo, useRef, useEffect, useCallback, Suspense } from "react";
import { createPortal } from "react-dom";
import { Search, Menu } from "lucide-react";
import { IOP_MODULES, type IopModuleDef } from "../modules/registry";
import { buildGroupTabs } from "./darkroomModules";
import {
  MODULE_GROUP_PRESETS,
  DEFAULT_PRESET_NAME,
  type ModuleGroupPreset,
} from "./moduleGroupPresets";
import ProcessingModuleCard from "../modules/ProcessingModuleCard";
import ScopeWidget from "./ScopeWidget";
import BauhausTooltip from "../controls/BauhausTooltip";
import { useDevelopStore } from "../../stores/developStore";

function getPreset(name: string): ModuleGroupPreset {
  return MODULE_GROUP_PRESETS.find((p) => p.name === name) ?? MODULE_GROUP_PRESETS[0];
}

export default function DarkroomRightSidebar() {
  const [presetName, setPresetName] = useState(DEFAULT_PRESET_NAME);
  const preset = useMemo(() => getPreset(presetName), [presetName]);
  const groupTabs = useMemo(() => buildGroupTabs(preset.groups), [preset]);

  const [activeTab, setActiveTab] = useState("active");
  const [searchQuery, setSearchQuery] = useState("");
  const [presetsOpen, setPresetsOpen] = useState(false);
  const btnRef = useRef<HTMLButtonElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);
  const [popupPos, setPopupPos] = useState({ top: 0, left: 0, minWidth: 0 });
  const historyItems = useDevelopStore((s) => s.historyItems);
  const focusModuleOp = useDevelopStore((s) => s.focusModuleOp);

  // React to focusModuleOp: switch to active tab so the module is visible
  useEffect(() => {
    if (focusModuleOp) {
      setActiveTab("active");
    }
  }, [focusModuleOp]);

  // Close popup on outside click or Escape
  useEffect(() => {
    if (!presetsOpen) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (
        btnRef.current && !btnRef.current.contains(target) &&
        popupRef.current && !popupRef.current.contains(target)
      ) {
        setPresetsOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPresetsOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [presetsOpen]);

  const togglePresets = useCallback(() => {
    setPresetsOpen((prev) => {
      if (!prev && btnRef.current) {
        const rect = btnRef.current.getBoundingClientRect();
        setPopupPos({ top: rect.bottom + 2, left: rect.right, minWidth: 200 });
      }
      return !prev;
    });
  }, []);

  const selectPreset = useCallback((name: string) => {
    setPresetName(name);
    setPresetsOpen(false);
    setActiveTab("active");
  }, []);

  // Find the active preset group's module list
  const activePresetGroup = useMemo(
    () => preset.groups.find((g) => g.id === activeTab),
    [preset, activeTab],
  );

  const filteredModules = useMemo(() => {
    let modules: IopModuleDef[] = IOP_MODULES;

    if (activeTab === "quick") {
      return [];
    } else if (activeTab === "active") {
      const historyOps = new Set(historyItems.map((h) => h.op));
      modules = modules.filter((m) => historyOps.has(m.op));
    } else if (activePresetGroup) {
      const opSet = new Set(activePresetGroup.modules);
      modules = modules.filter((m) => opSet.has(m.op));
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
  }, [activeTab, activePresetGroup, searchQuery, historyItems]);

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Histogram / Waveform scope */}
      <ScopeWidget />

      {/* Module group tabs */}
      <div className="darkroom-group-tabs">
        {groupTabs.map((group) => (
          <BauhausTooltip key={group.id} content={group.label}>
            <button
              className="darkroom-group-tab"
              data-active={activeTab === group.id}
              onClick={() => setActiveTab(group.id)}
            >
              {group.icon}
            </button>
          </BauhausTooltip>
        ))}
        <BauhausTooltip content="presets">
          <button
            ref={btnRef}
            className="darkroom-group-presets"
            data-active={presetsOpen}
            onClick={togglePresets}
          >
            <Menu size={14} />
          </button>
        </BauhausTooltip>
        {presetsOpen && createPortal(
          <div
            ref={popupRef}
            className="bauhaus-combo-popup"
            style={{ top: popupPos.top, left: popupPos.left, minWidth: popupPos.minWidth, transform: "translateX(-100%)" }}
          >
            {MODULE_GROUP_PRESETS.map((p) => (
              <div
                key={p.name}
                className="preset-option"
                data-selected={p.name === presetName}
                onClick={() => selectPreset(p.name)}
              >
                <span className="preset-check">{p.name === presetName ? "✓" : ""}</span>
                {p.name}
              </div>
            ))}
          </div>,
          document.body,
        )}
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
            />
          ))}
        </Suspense>
        {filteredModules.length === 0 && activeTab !== "quick" && (
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
