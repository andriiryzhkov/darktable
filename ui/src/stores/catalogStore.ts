import { create } from "zustand";
import { catalogQuery } from "../api/commands";
import type { ImageInfo } from "../types/protocol";

interface CatalogState {
  images: ImageInfo[];
  total: number;
  offset: number;
  limit: number;
  loading: boolean;
  selectedId: number | null;
  fetchPage: (offset?: number) => Promise<void>;
  selectImage: (id: number | null) => void;
  nextPage: () => Promise<void>;
  prevPage: () => Promise<void>;
}

const PAGE_SIZE = 50;

export const useCatalogStore = create<CatalogState>((set, get) => ({
  images: [],
  total: 0,
  offset: 0,
  limit: PAGE_SIZE,
  loading: false,
  selectedId: null,

  fetchPage: async (offset?: number) => {
    const currentOffset = offset ?? get().offset;
    set({ loading: true });
    try {
      const result = await catalogQuery(currentOffset, PAGE_SIZE);
      set({
        images: result.images,
        total: result.total,
        offset: result.offset,
        limit: result.limit,
        loading: false,
      });
    } catch (e) {
      console.error("catalog.query failed:", e);
      set({ loading: false });
    }
  },

  selectImage: (id) => set({ selectedId: id }),

  nextPage: async () => {
    const { offset, total } = get();
    if (offset + PAGE_SIZE < total) {
      await get().fetchPage(offset + PAGE_SIZE);
    }
  },

  prevPage: async () => {
    const { offset } = get();
    if (offset > 0) {
      await get().fetchPage(Math.max(0, offset - PAGE_SIZE));
    }
  },
}));
