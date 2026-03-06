import { create } from "zustand";
import { configGet, configSet } from "../api/commands";

type View = "lighttable" | "darkroom";

interface UIState {
  activeView: View;
  darkroomImgId: number | null;
  leftSidebarOpen: boolean;
  rightSidebarOpen: boolean;
  leftSidebarWidth: number;
  rightSidebarWidth: number;
  thumbnailSize: number;
  gridColumns: number;
  /** Target column count loaded from config, applied once container is measured */
  targetColumns: number;
  filmstripOpen: boolean;
  filmstripHeight: number;
  showGuides: boolean;
  guidesModuleOpen: boolean;
  darkroomBorderSize: number | null;

  setActiveView: (view: View) => void;
  setDarkroomImgId: (imgid: number) => void;
  toggleLeftSidebar: () => void;
  toggleRightSidebar: () => void;
  setLeftSidebarWidth: (w: number) => void;
  setRightSidebarWidth: (w: number) => void;
  setThumbnailSize: (size: number) => void;
  setGridColumns: (cols: number) => void;
  toggleFilmstrip: () => void;
  setFilmstripHeight: (h: number) => void;
  setShowGuides: (show: boolean) => void;
  setGuidesModuleOpen: (open: boolean) => void;
  loadFromConfig: () => void;
}

const SIDEBAR_MIN = 150;
const SIDEBAR_MAX = 400;
const clampSidebar = (w: number) => Math.max(SIDEBAR_MIN, Math.min(SIDEBAR_MAX, w));

export const useUIStore = create<UIState>((set, get) => ({
  activeView: "lighttable",
  darkroomImgId: null,
  leftSidebarOpen: true,
  rightSidebarOpen: true,
  leftSidebarWidth: 280,
  rightSidebarWidth: 280,
  thumbnailSize: 200,
  gridColumns: 0,
  targetColumns: 0,
  filmstripOpen: true,
  filmstripHeight: 100,
  showGuides: false,
  guidesModuleOpen: false,
  darkroomBorderSize: null,

  setActiveView: (view) => set({ activeView: view }),
  setDarkroomImgId: (imgid) => set({ darkroomImgId: imgid }),
  toggleLeftSidebar: () =>
    set((s) => ({ leftSidebarOpen: !s.leftSidebarOpen })),
  toggleRightSidebar: () =>
    set((s) => ({ rightSidebarOpen: !s.rightSidebarOpen })),
  setLeftSidebarWidth: (w) => set({ leftSidebarWidth: clampSidebar(w) }),
  setRightSidebarWidth: (w) => set({ rightSidebarWidth: clampSidebar(w) }),
  setThumbnailSize: (size) => {
    const clamped = Math.max(100, Math.min(400, size));
    set({ thumbnailSize: clamped });
  },
  setGridColumns: (cols) => {
    const { targetColumns, thumbnailSize, gridColumns } = get();
    // On first measure after config load, apply target column count
    if (targetColumns > 0 && cols > 0) {
      const containerWidth = cols * thumbnailSize;
      const newSize = Math.floor(containerWidth / targetColumns);
      const clamped = Math.max(100, Math.min(400, newSize));
      set({ gridColumns: cols, targetColumns: 0, thumbnailSize: clamped });
      return;
    }
    set({ gridColumns: cols });
    // Persist column count when it changes from user action
    if (cols > 0 && cols !== gridColumns) {
      configSet("plugins/lighttable/images_in_row", String(cols)).catch(() => {});
    }
  },
  toggleFilmstrip: () => set((s) => ({ filmstripOpen: !s.filmstripOpen })),
  setFilmstripHeight: (h) => set({ filmstripHeight: Math.max(60, Math.min(200, h)) }),
  setShowGuides: (show) => set({ showGuides: show }),
  setGuidesModuleOpen: (open) => set({ guidesModuleOpen: open }),

  loadFromConfig: () => {
    configGet("plugins/lighttable/images_in_row")
      .then(({ value }) => {
        const n = parseInt(value, 10);
        if (n > 0 && n <= 20) {
          set({ targetColumns: n });
        }
      })
      .catch(() => {});
    configGet("plugins/darkroom/ui/border_size")
      .then(({ value }) => {
        const n = parseInt(value, 10);
        if (n >= 0) set({ darkroomBorderSize: n });
      })
      .catch(() => {});
  },
}));
