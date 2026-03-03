import { create } from "zustand";

type View = "lighttable" | "darkroom";

interface UIState {
  activeView: View;
  leftSidebarOpen: boolean;
  rightSidebarOpen: boolean;
  thumbnailSize: number;
  gridColumns: number;

  setActiveView: (view: View) => void;
  toggleLeftSidebar: () => void;
  toggleRightSidebar: () => void;
  setThumbnailSize: (size: number) => void;
  setGridColumns: (cols: number) => void;
}

export const useUIStore = create<UIState>((set) => ({
  activeView: "lighttable",
  leftSidebarOpen: true,
  rightSidebarOpen: true,
  thumbnailSize: 200,
  gridColumns: 0,

  setActiveView: (view) => set({ activeView: view }),
  toggleLeftSidebar: () =>
    set((s) => ({ leftSidebarOpen: !s.leftSidebarOpen })),
  toggleRightSidebar: () =>
    set((s) => ({ rightSidebarOpen: !s.rightSidebarOpen })),
  setThumbnailSize: (size) =>
    set({ thumbnailSize: Math.max(100, Math.min(400, size)) }),
  setGridColumns: (cols) => set({ gridColumns: cols }),
}));
