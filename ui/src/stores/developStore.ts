import { create } from "zustand";
import {
  developOpen,
  developClose,
  developSetParams,
  developCommitParams,
  developGetParams,
  developRequestPreview,
  developSamplePixels,
  developGetHistory,
  developDeleteHistory,
  getPreviewFrame,
} from "../api/commands";
import { onServerEvent } from "../api/events";
import type { ExposureParams, SigmoidParams, DemosaicParams, RawprepareParams, ModuleInfo, HistoryItem, PixelSampleResult } from "../types/protocol";

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
  previewSrc: string | null; // URL for <img> fallback path
  frameData: Uint8Array | null; // raw BGRA pixels for WebGL rendering
  frontBuffer: number;
  sequence: number;
  exposureParams: ExposureParams | null;
  sigmoidParams: SigmoidParams | null;
  demosaicParams: DemosaicParams | null;
  rawprepareParams: RawprepareParams | null;
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
  /** Lightweight: setParams (preview_only) + render. Use during drag — skips history write. */
  applyParam: (op: string, params: Record<string, unknown>) => Promise<void>;
  /** Commit current module params to history after a preview_only drag. */
  commitParam: (op: string) => Promise<void>;
  /** Full: setParams + render + frame + history + params refetch. Use on drag end. */
  setModuleParam: (op: string, params: Record<string, unknown>) => Promise<void>;
  enableModule: (op: string, enabled: boolean) => Promise<void>;
  fetchHistory: () => Promise<void>;
  deleteHistory: () => Promise<void>;
  samplePixels: (x: number, y: number, w: number, h: number) => Promise<PixelSampleResult | null>;
  setZoom: (zoom: ZoomLevel) => void;
  setPan: (x: number, y: number) => void;
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
  exposureParams: null,
  sigmoidParams: null,
  demosaicParams: null,
  rawprepareParams: null,
  modules: [],
  historyItems: [],
  historyEnd: 0,
  loading: false,
  previewError: null,
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

      // Render preview (server processes synchronously, SHM is ready when this returns)
      const preview = await developRequestPreview(result.session_id);
      if (gen !== sessionGeneration) return;

      set({
        frontBuffer: preview.front_buffer,
        sequence: preview.sequence,
        previewWidth: preview.width,
        previewHeight: preview.height,
      });

      // Fetch the rendered frame
      await get().fetchFrame();

      // Fetch history stack and module params after first preview
      await get().fetchHistory();
      await get().fetchModuleParams("exposure");
      await get().fetchModuleParams("sigmoid");
      await get().fetchModuleParams("demosaic");
      await get().fetchModuleParams("rawprepare");
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
      sigmoidParams: null,
      demosaicParams: null,
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
    const { sessionId, frontBuffer, sequence } = get();
    if (!sessionId) return;
    const port = (window as unknown as Record<string, number>).__dt_frame_port;
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
        console.error("[fetchFrame] raw fetch failed:", e);
      }
    } else {
      // Fallback: IPC + base64 JPEG
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
      } else if (op === "sigmoid") {
        set({ sigmoidParams: result.params as unknown as SigmoidParams });
      } else if (op === "demosaic") {
        set({ demosaicParams: result.params as unknown as DemosaicParams });
      } else if (op === "rawprepare") {
        set({ rawprepareParams: result.params as unknown as RawprepareParams });
      }
    } catch (e) {
      console.error(`fetch ${op} params failed:`, e);
    }
  },

  applyParam: async (op: string, params: Record<string, unknown>) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      const t0 = performance.now();
      await developSetParams(sessionId, op, params, true);
      console.log(`[perf] applyParam IPC: ${(performance.now() - t0).toFixed(1)}ms`);
    } catch (e) {
      console.error(`apply ${op} param failed:`, e);
    }
  },

  commitParam: async (op: string) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developCommitParams(sessionId, op);
    } catch (e) {
      console.error(`commit ${op} params failed:`, e);
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

// --- Event-driven preview update ---
// Subscribe to server-pushed "develop.preview_ready" events.
// When the async pipeline finishes, the server writes SHM and pushes this event.
// We update the store's buffer info and fetch the frame.

interface PreviewReadyEvent {
  session_id: string;
  front_buffer: number;
  width: number;
  height: number;
  sequence: number;
}

onServerEvent("develop.preview_ready", (raw: unknown) => {
  const tEvent = performance.now();
  const data = raw as PreviewReadyEvent;
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
});
