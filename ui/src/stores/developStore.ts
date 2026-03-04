import { create } from "zustand";
import {
  developOpen,
  developClose,
  developSetParams,
  developRequestPreview,
  getPreviewFrame,
} from "../api/commands";
import type { ExposureParams, ModuleInfo } from "../types/protocol";

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
  loading: boolean;
  previewError: string | null;

  openSession: (imgid: number) => Promise<void>;
  closeSession: () => Promise<void>;
  requestPreview: () => Promise<void>;
  fetchFrame: () => Promise<void>;
  setExposure: (value: number) => Promise<void>;
  setBlack: (value: number) => Promise<void>;
  enableModule: (op: string, enabled: boolean) => Promise<void>;
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
  loading: false,
  previewError: null,

  openSession: async (imgid: number) => {
    // Bump generation — any in-flight operations for prior sessions become stale
    const gen = ++sessionGeneration;

    // Close previous session if any (fire-and-forget)
    const prevSession = get().sessionId;
    if (prevSession) {
      developClose(prevSession).catch(() => {});
    }

    set({ loading: true, imgid, previewError: null, sessionId: null, previewSrc: null, frameData: null });

    try {
      const result = await developOpen(imgid, PREVIEW_WIDTH, PREVIEW_HEIGHT);
      if (gen !== sessionGeneration) return; // stale — a newer open superseded us

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
      developClose(sessionId).catch(() => {});
    }
    set({
      sessionId: null,
      imgid: null,
      previewSrc: null,
      frameData: null,
      exposureParams: null,
      modules: [],
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

  setExposure: async (value: number) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developSetParams(sessionId, "exposure", { exposure: value });
      await get().requestPreview();
      await get().fetchFrame();
    } catch (e) {
      console.error("set exposure failed:", e);
    }
  },

  setBlack: async (value: number) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await developSetParams(sessionId, "exposure", { black: value });
      await get().requestPreview();
      await get().fetchFrame();
    } catch (e) {
      console.error("set black failed:", e);
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
    } catch (e) {
      console.error("enable module failed:", e);
    }
  },
}));
