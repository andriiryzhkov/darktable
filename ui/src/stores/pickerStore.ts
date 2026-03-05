import { create } from "zustand";

export type PickerMode = "point" | "area";

interface PickerRequest {
  id: string;
  module: string;
  mode: PickerMode;
}

/** Normalized box in image space (0-1) */
export interface PickerBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

const DEFAULT_BOX: PickerBox = { x: 0.05, y: 0.05, w: 0.9, h: 0.9 };

interface PickerState {
  active: PickerRequest | null;
  box: PickerBox;
  activate: (id: string, module: string, mode: PickerMode) => void;
  deactivate: (id?: string) => void;
  setBox: (box: PickerBox) => void;
}

export const usePickerStore = create<PickerState>((set, get) => ({
  active: null,
  box: { ...DEFAULT_BOX },

  activate: (id, module, mode) => {
    const current = get().active;
    if (current?.id === id) {
      set({ active: null });
    } else {
      set({ active: { id, module, mode }, box: { ...DEFAULT_BOX } });
    }
  },

  deactivate: (id) => {
    const current = get().active;
    if (!id || current?.id === id) {
      set({ active: null });
    }
  },

  setBox: (box) => set({ box }),
}));

// Global listener: deactivate picker on any pointerdown outside picker UI
document.addEventListener("pointerdown", (e) => {
  if (!usePickerStore.getState().active) return;
  const target = e.target as HTMLElement;
  if (target.closest(".picker-overlay") || target.closest(".bauhaus-picker")) return;
  usePickerStore.getState().deactivate();
});
