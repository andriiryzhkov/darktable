import { create } from "zustand";
import {
  developOpen,
  developClose,
  developSetParams,
  developGetParams,
  developRequestPreview,
  developGetHistory,
  developDeleteHistory,
  getPreviewFrame,
} from "../api/commands";
import type { ExposureParams, ModuleInfo, HistoryItem } from "../types/protocol";

export const ZOOM_LEVELS = ["small", "fit", "fill", "50", "100", "200", "400", "800", "1600"] as const;
export type ZoomLevel = (typeof ZOOM_LEVELS)[number];

export const ZOOM_LABELS: Record<ZoomLevel, string> = {
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

interface DevelopState {
  sessionId: string | null;
  imgid: number | null;
  previewWidth: number;
  previewHeight: number;
  previewSrc: string | null; // data URL for <img> (JPEG) or null
  frameData: Uint8Array | null; // raw BGRA fallback
  frontBuffer: number;
  sequence: number;
  exposureParams: ExposureParams | null;
  modules: ModuleInfo[];
  historyItems: HistoryItem[];
  historyEnd: number;
  loading: boolean;
  previewError: string | null;

  // Zoom & pan
  zoom: ZoomLevel;
  panX: number; // 0..1, center of viewport in image space
  panY: number;

  openSession: (imgid: number) => Promise<void>;
  closeSession: () => Promise<void>;
  requestPreview: () => Promise<void>;
  fetchFrame: () => Promise<void>;
  fetchModuleParams: (op: string) => Promise<void>;
  setModuleParam: (op: string, params: Record<string, unknown>) => Promise<void>;
  enableModule: (op: string, enabled: boolean) => Promise<void>;
  fetchHistory: () => Promise<void>;
  deleteHistory: () => Promise<void>;
  setZoom: (zoom: ZoomLevel) => void;
  setPan: (x: number, y: number) => void;
}

const PREVIEW_WIDTH = 1920;
const PREVIEW_HEIGHT = 1280;

// Generation counter to detect stale async operations
let sessionGeneration = 0;

export const useDevelopStore = create<DevelopState>((set, get) => ({
  sessionId: null,
  imgid: null,
  previewWidth: PREVIEW_WIDTH,
  previewHeight: PREVIEW_HEIGHT,
  previewSrc: null,
  frameData: null,
  frontBuffer: 0,
  sequence: 0,
  exposureParams: null,
  modules: [],
  historyItems: [],
  historyEnd: 0,
  loading: false,
  previewError: null,
  zoom: "fit" as ZoomLevel,
  panX: 0.5,
  panY: 0.5,

  setZoom: (zoom: ZoomLevel) => set({ zoom, panX: 0.5, panY: 0.5 }),
  setPan: (panX: number, panY: number) => set({ panX, panY }),

  openSession: async (imgid: number) => {
    // Bump generation — any in-flight operations for prior sessions become stale
    const gen = ++sessionGeneration;

    // Close previous session — must await so server frees the slot before we open a new one
    const prevSession = get().sessionId;
    if (prevSession) {
      await developClose(prevSession).catch(() => {});
    }

    set({ loading: true, imgid, previewError: null, sessionId: null, previewSrc: null, frameData: null, zoom: "fit" as ZoomLevel, panX: 0.5, panY: 0.5 });

    try {
      const result = await developOpen(imgid, PREVIEW_WIDTH, PREVIEW_HEIGHT);
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

      // Render preview (server processes synchronously, SHM is ready when this returns)
      const preview = await developRequestPreview(result.session_id);
      if (gen !== sessionGeneration) return;

      set({
        frontBuffer: preview.front_buffer,
        sequence: preview.sequence,
        previewWidth: preview.width,
        previewHeight: preview.height,
      });

      // Read the frame from SHM
      const frame = await getPreviewFrame(result.session_id, preview.front_buffer);
      if (gen !== sessionGeneration) return;

      if (frame.format === "jpeg") {
        set({
          previewSrc: `data:image/jpeg;base64,${frame.data}`,
          frameData: null,
          previewWidth: frame.width,
          previewHeight: frame.height,
          previewError: null,
        });
      } else {
        const binaryStr = atob(frame.data);
        const bytes = new Uint8Array(binaryStr.length);
        for (let i = 0; i < binaryStr.length; i++) {
          bytes[i] = binaryStr.charCodeAt(i);
        }
        set({
          previewSrc: null,
          frameData: bytes,
          previewWidth: frame.width,
          previewHeight: frame.height,
          previewError: null,
        });
      }

      // Fetch history stack and module params after first preview
      await get().fetchHistory();
      await get().fetchModuleParams("exposure");
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
      exposureParams: null,
      modules: [],
      historyItems: [],
      historyEnd: 0,
      sequence: 0,
    });
  },

  requestPreview: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      const result = await developRequestPreview(sessionId);
      set({
        frontBuffer: result.front_buffer,
        sequence: result.sequence,
        previewWidth: result.width,
        previewHeight: result.height,
      });
    } catch (e) {
      console.error("develop.request_preview failed:", e);
    }
  },

  fetchFrame: async () => {
    const { sessionId, frontBuffer } = get();
    if (!sessionId) return;
    const result = await getPreviewFrame(sessionId, frontBuffer);

    if (result.format === "jpeg") {
      set({
        previewSrc: `data:image/jpeg;base64,${result.data}`,
        frameData: null,
        previewWidth: result.width,
        previewHeight: result.height,
        previewError: null,
      });
    } else {
      const binaryStr = atob(result.data);
      const bytes = new Uint8Array(binaryStr.length);
      for (let i = 0; i < binaryStr.length; i++) {
        bytes[i] = binaryStr.charCodeAt(i);
      }
      set({
        previewSrc: null,
        frameData: bytes,
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

  fetchModuleParams: async (op: string) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      const result = await developGetParams(sessionId, op);
      if (op === "exposure") {
        set({ exposureParams: result.params as unknown as ExposureParams });
      }
    } catch (e) {
      console.error(`fetch ${op} params failed:`, e);
    }
  },

  setModuleParam: async (op: string, params: Record<string, unknown>) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developSetParams(sessionId, op, params);
      await get().requestPreview();
      await get().fetchFrame();
      await get().fetchHistory();
      await get().fetchModuleParams(op);
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
}));
