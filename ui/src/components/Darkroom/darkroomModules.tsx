import type { ReactNode } from "react";
import {
  SlidersHorizontal, Power, Circle, Sun, Palette,
  Wrench, Sparkles, SwatchBook,
} from "lucide-react";
import type { PresetGroup } from "./moduleGroupPresets";

export interface ModuleGroupDef {
  id: string;
  label: string;
  icon: ReactNode;
}

const S = 16;

const ICON_MAP: Record<string, ReactNode> = {
  Circle: <Circle size={S} />,
  Sun: <Sun size={S} />,
  Palette: <Palette size={S} />,
  Wrench: <Wrench size={S} />,
  Sparkles: <Sparkles size={S} />,
  SwatchBook: <SwatchBook size={S} />,
};

// Fixed tabs that always appear
export const FIXED_TABS: ModuleGroupDef[] = [
  { id: "quick", label: "quick access", icon: <SlidersHorizontal size={S} /> },
  { id: "active", label: "active modules", icon: <Power size={S} /> },
];

export function buildGroupTabs(presetGroups: PresetGroup[]): ModuleGroupDef[] {
  return [
    ...FIXED_TABS,
    ...presetGroups.map((g) => ({
      id: g.id,
      label: g.label,
      icon: ICON_MAP[g.icon] ?? <Circle size={S} />,
    })),
  ];
}
