import { create } from "zustand";

type View = "lighttable" | "darkroom";

interface UIState {
  activeView: View;
  leftSidebarOpen: boolean;
  rightSidebarOpen: boolean;
  leftSidebarWidth: number;
  rightSidebarWidth: number;
  thumbnailSize: number;
  gridColumns: number;

  setActiveView: (view: View) => void;
  toggleLeftSidebar: () => void;
  toggleRightSidebar: () => void;
  setLeftSidebarWidth: (w: number) => void;
  setRightSidebarWidth: (w: number) => void;
  setThumbnailSize: (size: number) => void;
  setGridColumns: (cols: number) => void;
}

const SIDEBAR_MIN = 150;
const SIDEBAR_MAX = 400;
const clampSidebar = (w: number) => Math.max(SIDEBAR_MIN, Math.min(SIDEBAR_MAX, w));

export const useUIStore = create<UIState>((set) => ({
  activeView: "lighttable",
  leftSidebarOpen: true,
  rightSidebarOpen: true,
  leftSidebarWidth: 280,
  rightSidebarWidth: 280,
  thumbnailSize: 200,
  gridColumns: 0,

  setActiveView: (view) => set({ activeView: view }),
  toggleLeftSidebar: () =>
    set((s) => ({ leftSidebarOpen: !s.leftSidebarOpen })),
  toggleRightSidebar: () =>
    set((s) => ({ rightSidebarOpen: !s.rightSidebarOpen })),
  setLeftSidebarWidth: (w) => set({ leftSidebarWidth: clampSidebar(w) }),
  setRightSidebarWidth: (w) => set({ rightSidebarWidth: clampSidebar(w) }),
  setThumbnailSize: (size) =>
    set({ thumbnailSize: Math.max(100, Math.min(400, size)) }),
  setGridColumns: (cols) => set({ gridColumns: cols }),
}));
