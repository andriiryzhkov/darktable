import { create } from "zustand";
import {
  developOpen,
  developClose,
  developSetParams,
  developCommitParams,
  developResetParams,
  developGetParams,
  developGetIntrospection,
  developRequestPreview,
  developSamplePixels,
  developGetModules,
  developGetHistory,
  developSelectHistory,
  developCompressHistory,
  developTruncateHistory,
  developDeleteHistory,
  developListPresets,
  developApplyPreset,
  developStorePreset,
  developDeletePreset,
  developNewInstance,
  developDeleteInstance,
  developMoveInstance,
  developRenameInstance,
  developGetMasks,
  developRenameMask,
  developDeleteMask,
  developCreateMask,
  developUpdateMask,
  developAssignMask,
  developSetBlendParam,
  developGetDistortionGrid,
  getPreviewFrame,
  getFramePort,
} from "../api/commands";
import { on } from "../events/eventBus";
import { useCatalogStore } from "./catalogStore";
import type { ModuleInfo, ModuleDescription, HistoryItem, PixelSampleResult, PresetInfo, IntrospectionResult, MaskForm, MaskUsage, DistortionGrid } from "../types/protocol";
import { getDragOps } from "../lib/maskDrag";

// Cached frame server port (resolved once, never changes)
let _cachedFramePort: number | undefined;

// Adaptive JPEG quality: lower quality during drag for faster feedback,
// full quality on release. Set to false to always use full quality.
const ADAPTIVE_JPEG = true;
const JPEG_QUALITY_FULL = 92;
const JPEG_QUALITY_INTERACTIVE = 60;

export const ZOOM_PRESETS = ["small", "fit", "fill", "50", "100", "200", "400", "800", "1600"] as const;
export type ZoomPreset = (typeof ZOOM_PRESETS)[number];
// ZoomLevel: named modes or any numeric percentage (50–1600)
export type ZoomLevel = ZoomPreset | number;
// Keep ZOOM_LEVELS as alias for combo box options
export const ZOOM_LEVELS = ZOOM_PRESETS;

export const ZOOM_LABELS: Record<ZoomPreset, string> = {
  small: "small",
  fit: "fit",
  fill: "fill",
  "50": "50%",
  "100": "100%",
  "200": "200%",
  "400": "400%",
  "800": "800%",
  "1600": "1600%",
};

const ZOOM_MIN = 50;
const ZOOM_MAX = 1600;

/** Get the numeric zoom factor (1.0 = 100%) for CSS transform */
export function getZoomFactor(zoom: ZoomLevel): number {
  if (typeof zoom === "number") return zoom / 100;
  return 1;
}

/** Check if zoom is a magnified/numeric mode (not fit/fill/small) */
export function isZoomedIn(zoom: ZoomLevel): boolean {
  return typeof zoom === "number";
}

/** Get display label for current zoom level */
export function getZoomLabel(zoom: ZoomLevel): string {
  if (typeof zoom === "number") {
    // Snap to preset label if close enough
    const preset = `${Math.round(zoom)}` as ZoomPreset;
    if (preset in ZOOM_LABELS) return ZOOM_LABELS[preset];
    return `${Math.round(zoom)}%`;
  }
  return ZOOM_LABELS[zoom] ?? String(zoom);
}

// Key levels that the scroll zoom snaps to when crossing (matches GTK darktable)
const SNAP_LEVELS = [50, 100, 200];
// Above 200%, GTK uses 2x jumps between these levels
const HIGH_ZOOM_STEPS = [200, 400, 800, 1600];

/**
 * Apply scroll zoom delta, matching GTK darktable behavior:
 * - Below 200%: continuous 1.1x per scroll step with snapping to key levels
 * - At/above 200%: discrete 2x jumps (200→400→800→1600)
 * - deltaY: raw wheel deltaY (positive = scroll down = zoom out)
 */
export function applyZoomDelta(zoom: ZoomLevel, deltaY: number): ZoomLevel {
  // Convert named modes to a starting percentage
  let current: number;
  if (typeof zoom === "number") {
    current = zoom;
  } else {
    current = ZOOM_MIN;
  }

  const zoomingIn = deltaY < 0;

  // Above 200%: discrete 2x steps (like GTK closeup levels)
  if (current >= 200) {
    // Only step on sufficient delta
    if (Math.abs(deltaY) < 20) return zoom;
    const idx = HIGH_ZOOM_STEPS.indexOf(current);
    if (idx >= 0) {
      const next = zoomingIn ? idx + 1 : idx - 1;
      if (next >= 0 && next < HIGH_ZOOM_STEPS.length) return HIGH_ZOOM_STEPS[next];
      return current;
    }
    // Between steps — snap to nearest
    const nearest = HIGH_ZOOM_STEPS.reduce((a, b) =>
      Math.abs(b - current) < Math.abs(a - current) ? b : a);
    return nearest;
  }

  // Below 200%: continuous with 1.1x per ~100px of delta (matches GTK's 10% per step)
  // 1.1^(1/100) ≈ 1.000953 per deltaY unit
  const factor = Math.pow(1.000953, -deltaY);
  let next = Math.round(current * factor);
  next = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, next));

  // Snap to key levels when crossing them
  for (const snap of SNAP_LEVELS) {
    if ((current < snap && next > snap) || (current > snap && next < snap)) {
      return snap;
    }
  }

  return next;
}

interface DevelopState {
  sessionId: string | null;
  imgid: number | null;
  previewWidth: number;
  previewHeight: number;
  previewSrc: string | null; // URL for <img> fallback path
  frameData: Uint8Array | null; // raw BGRA pixels for WebGL rendering
  frontBuffer: number;
  sequence: number;
  modules: ModuleInfo[];
  moduleDescriptions: Record<string, ModuleDescription>; // op → description from server
  genericParams: Record<string, Record<string, unknown>>; // op → params from introspection
  introspectionSchemas: Record<string, IntrospectionResult>; // op → schema cache
  historyItems: HistoryItem[];
  historyEnd: number;
  maskForms: MaskForm[];
  maskUsage: MaskUsage[];
  showMasks: boolean;
  selectedMaskId: number | null;
  /** Distortion grid for client-side mask coordinate transforms */
  distortionGrid: DistortionGrid | null;
  creationTool: "circle" | "ellipse" | "gradient" | "path" | "brush" | null;
  creationModule: { op: string; instance: number } | null;
  /** Form ID of mask currently being placed (follows cursor until clicked) */
  creatingMaskId: number | null;
  /** Brush creation settings shared between overlay and UI */
  brushSettings: { border: number; hardness: number; opacity: number; smoothing: "low" | "medium" | "high" };
  setBrushSettings: (settings: Partial<{ border: number; hardness: number; opacity: number; smoothing: "low" | "medium" | "high" }>) => void;
  loading: boolean;
  previewError: string | null;

  // Focus module (shift+click history item → scroll to & expand module in sidebar)
  focusModuleOp: string | null;

  // Interactive editing state (true during slider drag)
  interacting: boolean;

  // Zoom & pan
  zoom: ZoomLevel;
  panX: number; // 0..1, center of viewport in image space
  panY: number;

  openSession: (imgid: number) => Promise<void>;
  closeSession: () => Promise<void>;
  requestPreview: () => Promise<void>;
  fetchFrame: () => Promise<void>;
  fetchGenericParams: (op: string) => Promise<void>;
  /** Lightweight: setParams (preview_only) + render. Use during drag — skips history write. */
  applyParam: (op: string, params: Record<string, unknown>) => Promise<void>;
  /** Commit current module params to history after a preview_only drag. */
  commitParam: (op: string) => Promise<void>;
  /** Full: setParams + render + frame + history + params refetch. Use on drag end. */
  setModuleParam: (op: string, params: Record<string, unknown>) => Promise<void>;
  enableModule: (op: string, enabled: boolean) => Promise<void>;
  resetModule: (op: string) => Promise<void>;
  fetchHistory: () => Promise<void>;
  selectHistory: (historyEnd: number) => Promise<void>;
  compressHistory: () => Promise<void>;
  truncateHistory: (historyEnd: number) => Promise<void>;
  deleteHistory: () => Promise<void>;
  samplePixels: (x: number, y: number, w: number, h: number) => Promise<PixelSampleResult | null>;
  setZoom: (zoom: ZoomLevel) => void;
  setPan: (x: number, y: number) => void;
  focusModule: (op: string) => void;
  listPresets: (op: string) => Promise<PresetInfo[]>;
  applyPreset: (op: string, name: string) => Promise<void>;
  storePreset: (op: string, name: string, description?: string, filters?: import("../components/modules/StorePresetDialog").PresetFilterParams) => Promise<void>;
  removePreset: (op: string, name: string) => Promise<void>;
  newInstance: (op: string, instance?: number, copyParams?: boolean) => Promise<void>;
  deleteInstance: (op: string, instance: number) => Promise<void>;
  moveInstance: (op: string, instance: number, direction: "up" | "down") => Promise<void>;
  renameInstance: (op: string, instance: number, name: string) => Promise<void>;
  fetchIntrospection: (op: string) => Promise<IntrospectionResult | null>;
  fetchMasks: () => Promise<void>;
  fetchDistortionGrid: () => Promise<void>;
  renameMask: (formid: number, name: string) => Promise<void>;
  deleteMask: (formid: number) => Promise<void>;
  createMask: (type: "circle" | "ellipse" | "gradient" | "path" | "brush", params: Record<string, unknown>) => Promise<number | null>;
  updateMask: (formid: number, params: Record<string, unknown>) => Promise<void>;
  startCreation: (tool: "circle" | "ellipse" | "gradient" | "path" | "brush", op?: string, instance?: number) => void;
  resetCreation: () => void;
  /** Send lightweight mask update to server (no history write) and refresh polylines */
  previewMaskParam: (formid: number, updates: Record<string, unknown>) => void;
  /** Save mask creation (commit position, exit creation mode) */
  saveCreation: (position: [number, number]) => Promise<void>;
  /** Cancel mask creation and delete the form being created */
  cancelCreation: () => Promise<void>;
  assignMask: (formid: number, op: string, instance: number) => Promise<void>;
  clearModuleMasks: (op: string, instance: number) => Promise<void>;
  toggleMasks: () => void;
  selectMask: (formid: number | null) => void;
  setBlendParam: (op: string, instance: number, param: string, value: number, skipRefresh?: boolean) => Promise<void>;
}

/** Compute preview dimensions (CSS pixels, no DPR — pipeline cost scales with pixel count). */
function getPreviewDimensions() {
  const w = Math.min(Math.round(window.innerWidth * 0.7), 1920);
  const h = Math.min(window.innerHeight, 1200);
  return { width: Math.max(w, 640), height: Math.max(h, 480) };
}
// Generation counter to detect stale async operations
let sessionGeneration = 0;

/**
 * Derive enabled module ops from history items.
 * For each op, the last history entry determines if it's enabled.
 */
export function getEnabledOps(historyItems: HistoryItem[]): Set<string> {
  const last = new Map<string, boolean>();
  for (const item of historyItems) {
    last.set(item.op, item.enabled);
  }
  const enabled = new Set<string>();
  for (const [op, on] of last) {
    if (on) enabled.add(op);
  }
  return enabled;
}

export const useDevelopStore = create<DevelopState>((set, get) => ({
  sessionId: null,
  imgid: null,
  previewWidth: 0,
  previewHeight: 0,
  previewSrc: null,
  frameData: null,
  frontBuffer: 0,
  sequence: 0,
  modules: [],
  moduleDescriptions: {},
  genericParams: {},
  introspectionSchemas: {},
  historyItems: [],
  historyEnd: 0,
  maskForms: [],
  maskUsage: [],
  showMasks: false,
  selectedMaskId: null,
  distortionGrid: null,
  creationTool: null,
  creationModule: null,
  creatingMaskId: null,
  brushSettings: { border: 0.05, hardness: 0.5, opacity: 1.0, smoothing: "medium" as const },
  setBrushSettings: (s) => set((prev) => ({ brushSettings: { ...prev.brushSettings, ...s } })),
  loading: false,
  previewError: null,
  focusModuleOp: null,
  interacting: false,
  zoom: "fit" as ZoomLevel,
  panX: 0.5,
  panY: 0.5,

  samplePixels: async (x: number, y: number, w: number, h: number) => {
    const { sessionId } = get();
    if (!sessionId) return null;
    try {
      return await developSamplePixels(sessionId, x, y, w, h);
    } catch (e) {
      console.error("[develop] sample_pixels failed:", e);
      return null;
    }
  },

  setZoom: (zoom: ZoomLevel) => set({ zoom, panX: 0.5, panY: 0.5 }),
  setPan: (panX: number, panY: number) => set({ panX, panY }),
  focusModule: (op: string) => set({ focusModuleOp: op }),

  openSession: async (imgid: number) => {
    // Bump generation — any in-flight operations for prior sessions become stale
    const gen = ++sessionGeneration;

    // Close previous session — must await so server frees the slot before we open a new one
    const prevSession = get().sessionId;
    if (prevSession) {
      await developClose(prevSession).catch(() => {});
      // Bump thumbRevision so filmstrip re-fetches thumbnails for the edited image
      useCatalogStore.getState().bumpThumbRevision();
    }

    set({ loading: true, imgid, previewError: null, sessionId: null, previewSrc: null, frameData: null, zoom: "fit" as ZoomLevel, panX: 0.5, panY: 0.5 });

    try {
      const { width: pw, height: ph } = getPreviewDimensions();
      const result = await developOpen(imgid, pw, ph);
      if (gen !== sessionGeneration) {
        // Orphaned session — close it so server frees SHM buffers
        developClose(result.session_id).catch(() => {});
        return;
      }

      set({
        sessionId: result.session_id,
        previewWidth: result.preview_width,
        previewHeight: result.preview_height,
      });

      // Start history + module info + masks + params fetch in parallel with preview render
      const metadataPromise = Promise.all([
        get().fetchHistory(),
        get().fetchMasks(),
        (typeof window.developGetModules === "function"
          ? developGetModules(result.session_id).then((res) => {
              if (gen !== sessionGeneration) return;
              console.log("[developStore] get_modules response:", JSON.stringify(res).substring(0, 500));
              const descs: Record<string, ModuleDescription> = {};
              for (const m of res.modules) {
                if (m.description) descs[m.op] = m.description;
              }
              console.log("[developStore] moduleDescriptions:", Object.keys(descs));
              set({ modules: res.modules, moduleDescriptions: descs });
            })
          : Promise.resolve()
        ).catch((e) => console.error("develop.get_modules failed:", e)),
        get().fetchGenericParams("temperature"),
        get().fetchGenericParams("exposure"),
        get().fetchGenericParams("flip"),
        get().fetchGenericParams("sigmoid"),
        get().fetchGenericParams("demosaic"),
        get().fetchGenericParams("rawprepare"),
        get().fetchGenericParams("colorin"),
        get().fetchGenericParams("colorout"),
      ]);

      // Trigger async preview render — the develop.preview_ready event handler
      // will update the store and fetch the frame when the pipeline finishes.
      await developRequestPreview(result.session_id);
      if (gen !== sessionGeneration) return;

      // Wait for metadata to finish (preview will arrive via event)
      await metadataPromise;
    } catch (e) {
      if (gen !== sessionGeneration) return;
      const msg = e instanceof Error ? e.message : String(e);
      console.error("[develop] open failed:", msg);
      set({ previewError: msg });
    } finally {
      if (gen === sessionGeneration) {
        set({ loading: false });
      }
    }
  },

  closeSession: async () => {
    ++sessionGeneration; // invalidate any in-flight operations
    const { sessionId } = get();
    if (sessionId) {
      await developClose(sessionId).catch(() => {});
    }
    set({
      sessionId: null,
      imgid: null,
      previewSrc: null,
      frameData: null,
      modules: [],
      moduleDescriptions: {},
      historyItems: [],
      historyEnd: 0,
      maskForms: [],
      maskUsage: [],
      showMasks: false,
      selectedMaskId: null,
      creationTool: null,
      creationModule: null,
      sequence: 0,
      distortionGrid: null,
    });
  },

  requestPreview: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      // Async: server queues the render and responds immediately.
      // The actual frame arrives via the develop.preview_ready event handler.
      await developRequestPreview(sessionId);
    } catch (e) {
      console.error("develop.request_preview failed:", e);
    }
  },

  fetchDistortionGrid: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      const grid = await developGetDistortionGrid(sessionId);
      set({ distortionGrid: grid });
    } catch (e) {
      console.error("fetchDistortionGrid failed:", e);
    }
  },

  fetchFrame: async () => {
    const { sessionId, frontBuffer, sequence, interacting } = get();
    if (!sessionId) return;
    if (_cachedFramePort === undefined) _cachedFramePort = await getFramePort();
    const port = _cachedFramePort;
    const jpegQuality = (ADAPTIVE_JPEG && interacting) ? JPEG_QUALITY_INTERACTIVE : JPEG_QUALITY_FULL;
    if (port) {
      try {
        // Fetch raw BGRA pixels over HTTP — no JPEG encoding, no base64
        const t0 = performance.now();
        const resp = await fetch(
          `http://localhost:${port}/raw?s=${sessionId}&b=${frontBuffer}&seq=${sequence}`
        );
        if (!resp.ok) throw new Error(`frame server: ${resp.status}`);
        const t1 = performance.now();
        const w = parseInt(resp.headers.get("X-Width") || "0");
        const h = parseInt(resp.headers.get("X-Height") || "0");
        const buf = await resp.arrayBuffer();
        const t2 = performance.now();
        set({
          previewSrc: null,
          frameData: new Uint8Array(buf),
          previewWidth: w,
          previewHeight: h,
          previewError: null,
        });
        console.log(`[perf] fetchFrame: fetch=${(t1-t0).toFixed(1)}ms read=${(t2-t1).toFixed(1)}ms total=${(t2-t0).toFixed(1)}ms ${w}x${h}`);
      } catch (e) {
        console.error("[fetchFrame] raw fetch failed, falling back to JPEG:", e);
        // Fallback: use /frame (JPEG) endpoint with adaptive quality
        try {
          const resp = await fetch(
            `http://localhost:${port}/frame?s=${sessionId}&b=${frontBuffer}&q=${jpegQuality}`
          );
          if (!resp.ok) throw new Error(`frame server JPEG: ${resp.status}`);
          const blob = await resp.blob();
          const url = URL.createObjectURL(blob);
          set({
            previewSrc: url,
            frameData: null,
            previewError: null,
          });
        } catch (e2) {
          console.error("[fetchFrame] JPEG fallback also failed:", e2);
        }
      }
    } else {
      // Fallback: IPC + base64 JPEG (always full quality — no adaptive control via binding)
      const result = await getPreviewFrame(sessionId, frontBuffer);
      set({
        previewSrc: `data:image/jpeg;base64,${result.data}`,
        frameData: null,
        previewWidth: result.width,
        previewHeight: result.height,
        previewError: null,
      });
    }
  },

  fetchHistory: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      const result = await developGetHistory(sessionId);
      set({ historyItems: result.items, historyEnd: result.history_end });
    } catch (e) {
      console.error("develop.get_history failed:", e);
    }
  },

  selectHistory: async (historyEnd: number) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developSelectHistory(sessionId, historyEnd);
      set({ historyEnd });
      // Re-render preview and refresh module params (active modules change)
      await get().requestPreview();
      await Promise.all([
        get().fetchFrame(),
        get().fetchGenericParams("exposure"),
        get().fetchGenericParams("temperature"),
        get().fetchGenericParams("sigmoid"),
      ]);
    } catch (e) {
      console.error("develop.select_history failed:", e);
    }
  },

  compressHistory: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developCompressHistory(sessionId);
      await get().requestPreview();
      await get().fetchFrame();
      await get().fetchHistory();
    } catch (e) {
      console.error("develop.compress_history failed:", e);
    }
  },

  truncateHistory: async (historyEnd: number) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developTruncateHistory(sessionId, historyEnd);
      await get().requestPreview();
      await get().fetchFrame();
      await get().fetchHistory();
    } catch (e) {
      console.error("develop.truncate_history failed:", e);
    }
  },

  deleteHistory: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developDeleteHistory(sessionId);
      await get().requestPreview();
      await get().fetchFrame();
      await get().fetchHistory();
    } catch (e) {
      console.error("develop.delete_history failed:", e);
    }
  },

  fetchGenericParams: async (op: string) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      const result = await developGetParams(sessionId, op);
      set({ genericParams: { ...get().genericParams, [op]: result.params } });
    } catch (e) {
      console.error(`fetch ${op} params failed:`, e);
    }
  },

  applyParam: async (op: string, params: Record<string, unknown>) => {
    const { sessionId, historyItems } = get();
    if (!sessionId) return;
    if (!get().interacting) set({ interacting: true });
    try {
      // Auto-enable module if it's currently off
      if (!getEnabledOps(historyItems).has(op)) {
        await developSetParams(sessionId, op, { enabled: true });
        await get().fetchHistory();
      }
      const t0 = performance.now();
      await developSetParams(sessionId, op, params, true);
      console.log(`[perf] applyParam: ${(performance.now() - t0).toFixed(1)}ms`);
    } catch (e) {
      console.error(`apply ${op} param failed:`, e);
    }
  },

  commitParam: async (op: string) => {
    const { sessionId } = get();
    if (!sessionId) return;
    set({ interacting: false });
    try {
      await developCommitParams(sessionId, op);
    } catch (e) {
      console.error(`commit ${op} params failed:`, e);
    }
  },

  setModuleParam: async (op: string, params: Record<string, unknown>) => {
    const { sessionId, historyItems } = get();
    if (!sessionId) return;
    try {
      // Auto-enable module if it's currently off
      if (!getEnabledOps(historyItems).has(op)) {
        await developSetParams(sessionId, op, { enabled: true });
      }
      await developSetParams(sessionId, op, params);
      await get().requestPreview();
      await get().fetchFrame();
      await get().fetchHistory();
      await get().fetchGenericParams(op);
    } catch (e) {
      console.error(`set ${op} params failed:`, e);
    }
  },

  enableModule: async (op: string, enabled: boolean) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developSetParams(sessionId, op, { enabled });
      set((s) => ({
        modules: s.modules.map((m) =>
          m.op === op ? { ...m, enabled } : m
        ),
      }));
      await get().requestPreview();
      await get().fetchFrame();
      await get().fetchHistory();
    } catch (e) {
      console.error("enable module failed:", e);
    }
  },

  resetModule: async (op: string) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developResetParams(sessionId, op);
      await get().requestPreview();
      await get().fetchFrame();
      await get().fetchHistory();
      await get().fetchGenericParams(op);
    } catch (e) {
      console.error("reset module failed:", e);
    }
  },

  listPresets: async (op: string) => {
    const { sessionId } = get();
    if (!sessionId) return [];
    try {
      const result = await developListPresets(sessionId, op);
      return result.presets;
    } catch (e) {
      console.error("list presets failed:", e);
      return [];
    }
  },

  applyPreset: async (op: string, name: string) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developApplyPreset(sessionId, op, name);
      await get().requestPreview();
      await get().fetchFrame();
      await get().fetchHistory();
      await get().fetchGenericParams(op);
    } catch (e) {
      console.error("apply preset failed:", e);
    }
  },

  storePreset: async (op: string, name: string, description?: string, filters?: import("../components/modules/StorePresetDialog").PresetFilterParams) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developStorePreset(sessionId, op, name, description, filters);
    } catch (e) {
      console.error("store preset failed:", e);
    }
  },

  removePreset: async (op: string, name: string) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developDeletePreset(op, name);
    } catch (e) {
      console.error("delete preset failed:", e);
    }
  },

  newInstance: async (op: string, instance?: number, copyParams?: boolean) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developNewInstance(sessionId, op, instance, copyParams);
      await get().requestPreview();
      await Promise.all([
        get().fetchFrame(),
        get().fetchHistory(),
        developGetModules(sessionId).then((res) => {
          const descs: Record<string, ModuleDescription> = {};
          for (const m of res.modules) {
            if (m.description) descs[m.op] = m.description;
          }
          set({ modules: res.modules, moduleDescriptions: descs });
        }),
      ]);
    } catch (e) {
      console.error("new instance failed:", e);
    }
  },

  deleteInstance: async (op: string, instance: number) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developDeleteInstance(sessionId, op, instance);
      await get().requestPreview();
      await Promise.all([
        get().fetchFrame(),
        get().fetchHistory(),
        developGetModules(sessionId).then((res) => {
          const descs: Record<string, ModuleDescription> = {};
          for (const m of res.modules) {
            if (m.description) descs[m.op] = m.description;
          }
          set({ modules: res.modules, moduleDescriptions: descs });
        }),
      ]);
    } catch (e) {
      console.error("delete instance failed:", e);
    }
  },

  moveInstance: async (op: string, instance: number, direction: "up" | "down") => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developMoveInstance(sessionId, op, instance, direction);
      await get().requestPreview();
      await Promise.all([
        get().fetchFrame(),
        get().fetchHistory(),
        developGetModules(sessionId).then((res) => {
          const descs: Record<string, ModuleDescription> = {};
          for (const m of res.modules) {
            if (m.description) descs[m.op] = m.description;
          }
          set({ modules: res.modules, moduleDescriptions: descs });
        }),
      ]);
    } catch (e) {
      console.error("move instance failed:", e);
    }
  },

  renameInstance: async (op: string, instance: number, name: string) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developRenameInstance(sessionId, op, instance, name);
      await developGetModules(sessionId).then((res) => {
        const descs: Record<string, ModuleDescription> = {};
        for (const m of res.modules) {
          if (m.description) descs[m.op] = m.description;
        }
        set({ modules: res.modules, moduleDescriptions: descs });
      });
    } catch (e) {
      console.error("rename instance failed:", e);
    }
  },

  fetchIntrospection: async (op: string) => {
    const { sessionId, introspectionSchemas } = get();
    if (!sessionId) return null;
    if (introspectionSchemas[op]) return introspectionSchemas[op];
    try {
      const result = await developGetIntrospection(sessionId, op);
      set({ introspectionSchemas: { ...get().introspectionSchemas, [op]: result } });
      return result;
    } catch (e) {
      console.error(`fetch introspection for ${op} failed:`, e);
      return null;
    }
  },

  fetchMasks: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      const result = await developGetMasks(sessionId);
      set({ maskForms: result.forms, maskUsage: result.usage });
    } catch (e) {
      console.error("fetch masks failed:", e);
    }
  },

  renameMask: async (formid: number, name: string) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developRenameMask(sessionId, formid, name);
      get().fetchMasks();
    } catch (e) {
      console.error("rename mask failed:", e);
    }
  },

  deleteMask: async (formid: number) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developDeleteMask(sessionId, formid);
      get().fetchMasks();
      get().fetchHistory();
    } catch (e) {
      console.error("delete mask failed:", e);
    }
  },

  createMask: async (type, params) => {
    const { sessionId } = get();
    if (!sessionId) return null;
    // Clear any active creation tool (e.g. path drawing mode) when creating a mask
    if (get().creationTool) {
      set({ creationTool: null, creationModule: null });
    }
    try {
      const creation = !!(params as Record<string, unknown>)._creation;
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const { _creation, ...serverParams } = params as Record<string, unknown> & { _creation?: boolean };
      const result = await developCreateMask(sessionId, { type, ...serverParams });
      await get().fetchMasks();
      await get().fetchHistory();
      // Refresh modules to get updated blend params (mask_mode, mask_id)
      const mods = await developGetModules(sessionId);
      set({
        modules: mods.modules,
        selectedMaskId: result.formid,
        showMasks: true,
        creatingMaskId: creation ? result.formid : null,
      });
      return result.formid;
    } catch (e) {
      console.error("create mask failed:", e);
      return null;
    }
  },

  updateMask: async (formid, params) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developUpdateMask(sessionId, { formid, ...params });
      await get().fetchMasks();
    } catch (e) {
      console.error("update mask failed:", e);
    }
  },

  startCreation: (tool, op, instance) => {
    // Toggle off if already in this creation mode
    if (get().creationTool === tool) {
      const { creatingMaskId } = get();
      if (creatingMaskId) {
        get().deleteMask(creatingMaskId);
      }
      set({ creationTool: null, creationModule: null, creatingMaskId: null });
      return;
    }
    // Cancel any in-progress mask creation (e.g. circle/ellipse following cursor)
    const { creatingMaskId } = get();
    if (creatingMaskId) {
      set({ creatingMaskId: null, selectedMaskId: null });
      get().deleteMask(creatingMaskId);
    }
    set({ creationTool: tool, creationModule: op ? { op, instance: instance ?? 0 } : null, showMasks: true });
  },

  resetCreation: () => {
    set({ creationTool: null, creationModule: null });
  },

  previewMaskParam: (() => {
    // Coalescing state — only the latest update matters, skip intermediate ones
    let busy = false;
    let pending: { formid: number; updates: Record<string, unknown> } | null = null;
    let stopped = false;
    let idleResolvers: (() => void)[] = [];

    const fn = (formid: number, updates: Record<string, unknown>) => {
      if (stopped) return;
      // Update store immediately for instant bidirectional feedback (sliders ↔ canvas)
      const forms = get().maskForms;
      const updated = [...forms];
      let changed = false;
      // Update form geometry (center, radius, border, rotation, etc.)
      const { opacity: _opacity, ...geomUpdates } = updates;
      if (Object.keys(geomUpdates).length > 0) {
        const idx = updated.findIndex((f) => f.formid === formid);
        if (idx >= 0) {
          // Path masks: points is an array, replace entirely instead of spreading
          const newPoints = Array.isArray(geomUpdates.points)
            ? geomUpdates.points as unknown as MaskForm["points"]
            : { ...(updated[idx].points as unknown as Record<string, unknown>), ...geomUpdates } as unknown as MaskForm["points"];
          updated[idx] = { ...updated[idx], points: newPoints, transformed: undefined };
          changed = true;
        }
      }
      // Update opacity on group children referencing this form
      if (typeof _opacity === "number") {
        for (let i = 0; i < updated.length; i++) {
          if (!updated[i].children) continue;
          const ci = updated[i].children!.findIndex((c) => c.formid === formid);
          if (ci >= 0) {
            const newChildren = [...updated[i].children!];
            newChildren[ci] = { ...newChildren[ci], opacity: _opacity };
            updated[i] = { ...updated[i], children: newChildren };
            changed = true;
          }
        }
      }
      if (changed) set({ maskForms: updated });
      pending = { formid, updates };
      if (busy) return;
      busy = true;
      (async () => {
        try {
          const { sessionId } = get();
          if (!sessionId) return;
          while (pending && !stopped) {
            const { formid: fid, updates: upd } = pending;
            pending = null;
            await developUpdateMask(sessionId, { formid: fid, ...upd, preview_only: true });
          }
        } finally {
          busy = false;
          for (const r of idleResolvers) r();
          idleResolvers = [];
        }
      })();
    };

    // Stop accepting new requests and wait for in-flight request to complete
    fn.drain = async () => {
      stopped = true;
      pending = null;
      if (busy) await new Promise<void>(r => idleResolvers.push(r));
      stopped = false;
    };

    return fn;
  })(),

  saveCreation: async (position) => {
    const { creatingMaskId, sessionId } = get();
    if (!creatingMaskId || !sessionId) return;
    const form = get().maskForms.find((f) => f.formid === creatingMaskId);
    if (!form) return;
    const ops = getDragOps(form);
    if (!ops) return;
    // Set final position and collect all geometry params
    ops.setPosition(form, position);
    const params = ops.commitParams(form);
    try {
      // Stop background preview sync and wait for in-flight to complete,
      // preventing race where preview_only overwrites our committed position
      await (get().previewMaskParam as { drain: () => Promise<void> }).drain();
      // Commit final position (writes history, unlike preview_only)
      await developUpdateMask(sessionId, { formid: creatingMaskId, ...params });
      await get().fetchMasks();
      set({ creatingMaskId: null, selectedMaskId: creatingMaskId });
      get().requestPreview();
    } catch (e) {
      console.error("save creation failed:", e);
      set({ creatingMaskId: null });
    }
  },

  cancelCreation: async () => {
    const { creatingMaskId } = get();
    if (!creatingMaskId) return;
    set({ creatingMaskId: null, selectedMaskId: null });
    get().deleteMask(creatingMaskId);
  },

  assignMask: async (formid, op, instance) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developAssignMask(sessionId, { formid, op, instance });
      await get().fetchMasks();
      const result = await developGetModules(sessionId);
      set({ modules: result.modules });
    } catch (e) {
      console.error("assign mask failed:", e);
    }
  },

  clearModuleMasks: async (op, instance) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developSetBlendParam(sessionId, op, instance, "mask_id", 0);
      await get().fetchMasks();
      const result = await developGetModules(sessionId);
      set({ modules: result.modules });
    } catch (e) {
      console.error("clear module masks failed:", e);
    }
  },

  toggleMasks: () => set((s) => ({ showMasks: !s.showMasks })),
  selectMask: (formid) => set({ selectedMaskId: formid }),

  setBlendParam: async (op, instance, param, value, skipRefresh) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developSetBlendParam(sessionId, op, instance, param, value);
      if (!skipRefresh) {
        // Refresh modules to get updated blend params
        const result = await developGetModules(sessionId);
        set({ modules: result.modules });
      }
    } catch (e) {
      console.error("set blend param failed:", e);
    }
  },

}));

// --- Event-driven preview update ---
// Subscribe to server-pushed "develop.preview_ready" events.
// When the async pipeline finishes, the server writes SHM and pushes this event.
// We update the store's buffer info and fetch the frame.

on("develop.preview_ready", (data) => {
  const tEvent = performance.now();
  const state = useDevelopStore.getState();

  // Ignore events for other sessions
  if (data.session_id !== state.sessionId) return;

  // Ignore stale events (sequence must advance)
  if (data.sequence <= state.sequence) return;

  console.log(`[perf] preview_ready event: seq=${data.sequence} ${data.width}x${data.height}`);

  useDevelopStore.setState({
    frontBuffer: data.front_buffer,
    sequence: data.sequence,
    previewWidth: data.width,
    previewHeight: data.height,
  });

  // Fetch the frame pixels from SHM
  state.fetchFrame().then(() => {
    console.log(`[perf] event→render: ${(performance.now() - tEvent).toFixed(1)}ms`);
  });

  // Re-fetch masks now that the pipeline has completed — this enables
  // gui_points (distortion-transformed coordinates) which require pipe dimensions.
  state.fetchMasks();

  // Refresh distortion grid (used for client-side mask polyline computation)
  state.fetchDistortionGrid();
});

// Debug: expose store on window for runtime inspection
if (typeof window !== "undefined") {
  (window as unknown as Record<string, unknown>).__developStore = useDevelopStore;
}
