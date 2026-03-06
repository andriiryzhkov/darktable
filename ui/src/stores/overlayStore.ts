import { create } from "zustand";
import { OverlayMode, ThumbTableMode } from "../components/ThumbTable/types";
import { configGet, configSet } from "../api/commands";

/** Config key pattern matches darktable: plugins/lighttable/overlays/{mode}/{size} */
function overlayKey(mode: ThumbTableMode): string {
  // We use size=0 since our UI doesn't bucket by thumb size
  return `plugins/lighttable/overlays/${mode}/0`;
}
function tooltipKey(mode: ThumbTableMode): string {
  return `plugins/lighttable/tooltips/${mode}/0`;
}
function timeoutKey(mode: ThumbTableMode): string {
  return `plugins/lighttable/overlays_block_timeout/${mode}/0`;
}

interface OverlayState {
  /** Per-mode overlay settings */
  modes: Record<ThumbTableMode, {
    overlay: OverlayMode;
    tooltip: boolean;
    blockTimeout: number;
  }>;
  setOverlay: (mode: ThumbTableMode, overlay: OverlayMode) => void;
  setTooltip: (mode: ThumbTableMode, tooltip: boolean) => void;
  setBlockTimeout: (mode: ThumbTableMode, timeout: number) => void;
  loadFromConfig: () => void;
}

const DEFAULTS: OverlayState["modes"] = {
  [ThumbTableMode.Filemanager]: { overlay: OverlayMode.Mixed, tooltip: false, blockTimeout: 2 },
  [ThumbTableMode.Filmstrip]: { overlay: OverlayMode.HoverNormal, tooltip: false, blockTimeout: 2 },
};

export const useOverlayStore = create<OverlayState>((set) => ({
  modes: { ...DEFAULTS },

  setOverlay: (mode, overlay) => {
    set((s) => ({
      modes: { ...s.modes, [mode]: { ...s.modes[mode], overlay } },
    }));
    configSet(overlayKey(mode), String(overlay)).catch(() => {});
  },

  setTooltip: (mode, tooltip) => {
    set((s) => ({
      modes: { ...s.modes, [mode]: { ...s.modes[mode], tooltip } },
    }));
    configSet(tooltipKey(mode), tooltip ? "TRUE" : "FALSE").catch(() => {});
  },

  setBlockTimeout: (mode, timeout) => {
    set((s) => ({
      modes: { ...s.modes, [mode]: { ...s.modes[mode], blockTimeout: timeout } },
    }));
    configSet(timeoutKey(mode), String(timeout)).catch(() => {});
  },

  loadFromConfig: () => {
    for (const mode of [ThumbTableMode.Filemanager, ThumbTableMode.Filmstrip]) {
      configGet(overlayKey(mode))
        .then(({ value }) => {
          const n = parseInt(value, 10);
          if (n >= 0 && n <= 6) {
            set((s) => ({
              modes: { ...s.modes, [mode]: { ...s.modes[mode], overlay: n as OverlayMode } },
            }));
          }
        })
        .catch(() => {});
      configGet(tooltipKey(mode))
        .then(({ value }) => {
          set((s) => ({
            modes: { ...s.modes, [mode]: { ...s.modes[mode], tooltip: value === "TRUE" } },
          }));
        })
        .catch(() => {});
      configGet(timeoutKey(mode))
        .then(({ value }) => {
          const n = parseInt(value, 10);
          if (n > 0) {
            set((s) => ({
              modes: { ...s.modes, [mode]: { ...s.modes[mode], blockTimeout: n } },
            }));
          }
        })
        .catch(() => {});
    }
  },
}));
