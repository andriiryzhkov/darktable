import { create } from "zustand";
import { emit } from "../events/eventBus";
import type { CollectionRuleParam } from "../types/collections";

export type FilterType = "module_order" | "color_label" | "range_rating";

export const FILTER_TYPE_LABELS: Record<FilterType, string> = {
  module_order: "module order",
  color_label: "color label",
  range_rating: "range rating",
};

export const ALL_FILTER_TYPES: FilterType[] = ["module_order", "color_label", "range_rating"];

export type ColorMode = "union" | "intersection";

export const ALL_COLOR_KEYS = ["red", "yellow", "green", "blue", "purple"] as const;


export const MODULE_ORDER_OPTIONS = [
  { value: "all", label: "all images" },
  { value: "custom", label: "custom" },
  { value: "legacy", label: "legacy" },
  { value: "v3.0_raw", label: "v3.0 RAW" },
  { value: "v3.0_jpeg", label: "v3.0 JPEG" },
  { value: "v5.0_raw", label: "v5.0 RAW" },
  { value: "v5.0_jpeg", label: "v5.0 JPEG" },
  { value: "none", label: "none" },
] as const;

export type SortDirection = "asc" | "desc";

export const SORT_GROUPS = [
  {
    label: "files",
    options: ["filename", "full path", "aspect ratio"],
  },
  {
    label: "times",
    options: ["capture time", "import time", "modification time", "export time", "print time"],
  },
  {
    label: "metadata",
    options: ["rating", "color label", "title", "description"],
  },
  {
    label: "darktable",
    options: ["group", "id", "custom sort", "shuffle"],
  },
] as const;

interface FilterState {
  shownFilters: FilterType[];
  moduleOrder: string;
  activeColors: Set<string>;
  colorMode: ColorMode;
  ratingSelection: Set<number>; // -1=rejected, 0=unrated, 1-5=stars
  sortBy: string;
  sortDirection: SortDirection;
  grouping: boolean;

  setModuleOrder: (value: string) => void;
  toggleColor: (color: string) => void;
  toggleAllColors: () => void;
  toggleColorMode: () => void;
  clearColors: () => void;
  toggleRating: (value: number) => void;
  toggleRatingInSet: (value: number) => void;
  setRatingRange: (from: number, to: number) => void;
  clearRating: () => void;
  setSortBy: (value: string) => void;
  toggleSortDirection: () => void;
  toggleGrouping: () => void;
  addFilter: (type: FilterType) => void;
  removeFilter: (type: FilterType) => void;
  resetFilters: () => void;
  getFilterRules: () => CollectionRuleParam[];
}

export const useFilterStore = create<FilterState>((set, get) => ({
  shownFilters: ["module_order", "color_label", "range_rating"],
  moduleOrder: "all",
  activeColors: new Set<string>(),
  colorMode: "union",
  ratingSelection: new Set<number>(),
  sortBy: "filename",
  sortDirection: "asc",
  grouping: true,

  setModuleOrder: (value) => {
    set({ moduleOrder: value });
    if (value !== "all") {
      emit("collection.changed", { reason: "quick-filter" });
    }
  },

  toggleColor: (color) => {
    set((s) => {
      const next = new Set(s.activeColors);
      if (next.has(color)) next.delete(color);
      else next.add(color);
      return { activeColors: next };
    });
    emit("collection.changed", { reason: "quick-filter" });
  },

  toggleAllColors: () => {
    set((s) => {
      const allActive = ALL_COLOR_KEYS.every((k) => s.activeColors.has(k));
      return { activeColors: allActive ? new Set<string>() : new Set<string>(ALL_COLOR_KEYS) };
    });
    emit("collection.changed", { reason: "quick-filter" });
  },

  toggleColorMode: () => {
    set((s) => ({ colorMode: s.colorMode === "union" ? "intersection" : "union" }));
    if (get().activeColors.size > 0) {
      emit("collection.changed", { reason: "quick-filter" });
    }
  },

  clearColors: () => {
    set({ activeColors: new Set() });
    emit("collection.changed", { reason: "quick-filter" });
  },

  toggleRating: (value) => {
    set((s) => {
      // If this is the only selected value, deselect it; otherwise select only this one
      if (s.ratingSelection.size === 1 && s.ratingSelection.has(value)) {
        return { ratingSelection: new Set<number>() };
      }
      return { ratingSelection: new Set([value]) };
    });
    emit("collection.changed", { reason: "quick-filter" });
  },

  toggleRatingInSet: (value) => {
    set((s) => {
      const next = new Set(s.ratingSelection);
      if (next.has(value)) next.delete(value);
      else next.add(value);
      return { ratingSelection: next };
    });
    emit("collection.changed", { reason: "quick-filter" });
  },

  setRatingRange: (from, to) => {
    const lo = Math.min(from, to);
    const hi = Math.max(from, to);
    const next = new Set<number>();
    for (let i = lo; i <= hi; i++) next.add(i);
    set({ ratingSelection: next });
    emit("collection.changed", { reason: "quick-filter" });
  },

  clearRating: () => {
    set({ ratingSelection: new Set() });
    emit("collection.changed", { reason: "quick-filter" });
  },

  setSortBy: (value) => {
    set({ sortBy: value });
    emit("collection.changed", { reason: "sort" });
  },

  toggleSortDirection: () => {
    set((s) => ({ sortDirection: s.sortDirection === "asc" ? "desc" : "asc" }));
    emit("collection.changed", { reason: "sort" });
  },

  toggleGrouping: () => {
    set((s) => ({ grouping: !s.grouping }));
  },

  addFilter: (type) => {
    set((s) => {
      if (s.shownFilters.includes(type)) return s;
      return { shownFilters: [...s.shownFilters, type] };
    });
  },

  removeFilter: (type) => {
    set((s) => ({
      shownFilters: s.shownFilters.filter((f) => f !== type),
    }));
  },

  resetFilters: () => {
    set({
      shownFilters: ["module_order", "color_label", "range_rating"],
      moduleOrder: "all",
      activeColors: new Set(),
      colorMode: "union",
      ratingSelection: new Set(),
      sortBy: "filename",
      sortDirection: "asc",
    });
    emit("collection.changed", { reason: "quick-filter" });
  },

  getFilterRules: () => {
    const { moduleOrder, activeColors, colorMode, ratingSelection, shownFilters } = get();
    const rules: CollectionRuleParam[] = [];

    if (shownFilters.includes("module_order") && moduleOrder !== "all") {
      rules.push({
        mode: "and",
        property: "module_order",
        text: moduleOrder,
      });
    }

    if (shownFilters.includes("color_label") && activeColors.size > 0) {
      const join = colorMode === "intersection" ? "+" : ",";
      rules.push({
        mode: "and",
        property: "color_label",
        text: [...activeColors].join(join),
      });
    }

    if (shownFilters.includes("range_rating") && ratingSelection.size > 0) {
      // Build comma-separated list: rejected, unrated, and/or star values
      const parts: string[] = [];
      if (ratingSelection.has(-1)) parts.push("rejected");
      if (ratingSelection.has(0)) parts.push("unrated");
      const stars = [...ratingSelection].filter((v) => v >= 1).sort();
      for (const s of stars) parts.push(String(s));
      rules.push({
        mode: "and",
        property: "rating",
        text: parts.join(","),
      });
    }

    return rules;
  },
}));
