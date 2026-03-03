import { create } from "zustand";
import { catalogQuery } from "../api/commands";
import type { ImageInfo } from "../types/protocol";
import { on } from "../events/eventBus";
import { useCollectionsStore } from "./collectionsStore";

// Seed-based pseudo-random for deterministic mock data per image ID
function mockRng(seed: number): number {
  const x = Math.sin(seed * 9301 + 49297) * 233280;
  return x - Math.floor(x);
}

function enrichImage(img: ImageInfo): ImageInfo {
  return {
    ...img,
    rating: img.rating ?? Math.floor(mockRng(img.id) * 6),
    color_labels: img.color_labels ?? (mockRng(img.id + 1000) > 0.6
      ? Math.floor(mockRng(img.id + 2000) * 31)
      : 0),
    group_id: img.group_id ?? 0,
    altered: img.altered ?? mockRng(img.id + 3000) > 0.7,
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
      const rulesParams = useCollectionsStore.getState().getRulesParams();
      const rules = rulesParams.length > 0 ? rulesParams : undefined;
      // First query to discover total count
      const first = await catalogQuery(0, 1, rules);
      const total = first.total;
      if (total === 0) {
        set({ images: [], total: 0, loading: false });
        return;
      }
      // Fetch all image metadata in one request
      const result = await catalogQuery(0, total, rules);
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
}));

// React to events — reload collection when it changes
on("collection.changed", () => {
  useCatalogStore.getState().fetchAll();
});
on("import.finished", ({ imported }) => {
  if (imported > 0) useCatalogStore.getState().fetchAll();
});
