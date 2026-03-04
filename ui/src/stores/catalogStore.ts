import { create } from "zustand";
import { catalogQuery } from "../api/commands";
import type { ImageInfo } from "../types/protocol";
import { on } from "../events/eventBus";
import { useCollectionsStore } from "./collectionsStore";
import { useFilterStore } from "./filterStore";

function enrichImage(img: ImageInfo): ImageInfo {
  return {
    ...img,
    rating: img.rating ?? 0,
    color_labels: img.color_labels ?? 0,
    group_id: img.group_id ?? 0,
    altered: img.altered ?? false,
    local_copy: img.local_copy ?? false,
  };
}

interface CatalogState {
  images: ImageInfo[];
  total: number;
  loading: boolean;
  selectedIds: Set<number>;
  lastSelectedId: number | null;

  fetchAll: () => Promise<void>;
  selectImage: (id: number, e?: { ctrlKey?: boolean; metaKey?: boolean; shiftKey?: boolean }) => void;
  clearSelection: () => void;
  selectAll: () => void;
  invertSelection: () => void;
  selectFilmRoll: () => void;
  selectUntouched: () => void;
}

export const useCatalogStore = create<CatalogState>((set, get) => ({
  images: [],
  total: 0,
  loading: false,
  selectedIds: new Set<number>(),
  lastSelectedId: null,

  fetchAll: async () => {
    set({ loading: true });
    try {
      const collectionRules = useCollectionsStore.getState().getRulesParams();
      const filterRules = useFilterStore.getState().getFilterRules();
      const allRules = [...collectionRules, ...filterRules];
      const rules = allRules.length > 0 ? allRules : undefined;
      const { sortBy, sortDirection } = useFilterStore.getState();
      // First query to discover total count
      const first = await catalogQuery(0, 1, rules, sortBy, sortDirection);
      const total = first.total;
      if (total === 0) {
        set({ images: [], total: 0, loading: false });
        return;
      }
      // Fetch all image metadata in one request
      const result = await catalogQuery(0, total, rules, sortBy, sortDirection);
      set({
        images: result.images.map(enrichImage),
        total: result.total,
        loading: false,
      });
    } catch (e) {
      console.error("catalog.query failed:", e);
      set({ loading: false });
    }
  },

  selectImage: (id, e) => {
    const state = get();
    const multi = e?.ctrlKey || e?.metaKey;
    const range = e?.shiftKey;

    if (range && state.lastSelectedId !== null) {
      // Shift+click: select range
      const ids = state.images.map((img) => img.id);
      const fromIdx = ids.indexOf(state.lastSelectedId);
      const toIdx = ids.indexOf(id);
      if (fromIdx >= 0 && toIdx >= 0) {
        const start = Math.min(fromIdx, toIdx);
        const end = Math.max(fromIdx, toIdx);
        const newSelection = new Set(state.selectedIds);
        for (let i = start; i <= end; i++) {
          newSelection.add(ids[i]);
        }
        set({ selectedIds: newSelection });
        return;
      }
    }

    if (multi) {
      // Ctrl/Cmd+click: toggle
      const newSelection = new Set(state.selectedIds);
      if (newSelection.has(id)) {
        newSelection.delete(id);
      } else {
        newSelection.add(id);
      }
      set({ selectedIds: newSelection, lastSelectedId: id });
    } else {
      // Single click
      set({ selectedIds: new Set([id]), lastSelectedId: id });
    }
  },

  clearSelection: () => set({ selectedIds: new Set(), lastSelectedId: null }),

  selectAll: () => {
    const { images } = get();
    set({ selectedIds: new Set(images.map((img) => img.id)), lastSelectedId: null });
  },

  invertSelection: () => {
    const { images, selectedIds } = get();
    const inverted = new Set<number>();
    for (const img of images) {
      if (!selectedIds.has(img.id)) inverted.add(img.id);
    }
    set({ selectedIds: inverted, lastSelectedId: null });
  },

  selectFilmRoll: () => {
    const { images, selectedIds } = get();
    // Find film_id(s) of currently selected images, then select all images from those film rolls
    const selectedFilmIds = new Set<number>();
    for (const img of images) {
      if (selectedIds.has(img.id) && img.film_id != null) {
        selectedFilmIds.add(img.film_id);
      }
    }
    if (selectedFilmIds.size === 0) return;
    const newSelection = new Set<number>();
    for (const img of images) {
      if (img.film_id != null && selectedFilmIds.has(img.film_id)) {
        newSelection.add(img.id);
      }
    }
    set({ selectedIds: newSelection, lastSelectedId: null });
  },

  selectUntouched: () => {
    const { images } = get();
    const untouched = new Set<number>();
    for (const img of images) {
      if (!img.altered) untouched.add(img.id);
    }
    set({ selectedIds: untouched, lastSelectedId: null });
  },
}));

// React to events — reload collection when it changes
on("collection.changed", () => {
  useCatalogStore.getState().fetchAll();
});
on("import.finished", ({ imported }) => {
  if (imported > 0) useCatalogStore.getState().fetchAll();
});
