import { create } from "zustand";
import {
  developOpen,
  developClose,
  developSetParams,
  developRequestPreview,
  getPreviewFrame,
} from "../api/commands";
import type { ExposureParams } from "../types/protocol";

interface DevelopState {
  sessionId: string | null;
  imgid: number | null;
  previewWidth: number;
  previewHeight: number;
  frameData: Uint8Array | null;
  frontBuffer: number;
  sequence: number;
  exposureParams: ExposureParams | null;
  loading: boolean;

  openSession: (imgid: number) => Promise<void>;
  closeSession: () => Promise<void>;
  requestPreview: () => Promise<void>;
  fetchFrame: () => Promise<void>;
  setExposure: (value: number) => Promise<void>;
  setBlack: (value: number) => Promise<void>;
}

const PREVIEW_WIDTH = 1920;
const PREVIEW_HEIGHT = 1280;

export const useDevelopStore = create<DevelopState>((set, get) => ({
  sessionId: null,
  imgid: null,
  previewWidth: PREVIEW_WIDTH,
  previewHeight: PREVIEW_HEIGHT,
  frameData: null,
  frontBuffer: 0,
  sequence: 0,
  exposureParams: null,
  loading: false,

  openSession: async (imgid: number) => {
    set({ loading: true, imgid });
    try {
      const result = await developOpen(imgid, PREVIEW_WIDTH, PREVIEW_HEIGHT);
      set({
        sessionId: result.session_id,
        previewWidth: result.preview_width,
        previewHeight: result.preview_height,
      });

      // Render initial preview
      await get().requestPreview();
      await get().fetchFrame();
    } catch (e) {
      console.error("develop.open failed:", e);
    } finally {
      set({ loading: false });
    }
  },

  closeSession: async () => {
    const { sessionId } = get();
    if (sessionId) {
      try {
        await developClose(sessionId);
      } catch (e) {
        console.error("develop.close failed:", e);
      }
    }
    set({
      sessionId: null,
      imgid: null,
      frameData: null,
      exposureParams: null,
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
    try {
      const pixels = await getPreviewFrame(sessionId, frontBuffer);
      // Tauri returns Vec<u8> as number[] — convert to Uint8Array
      set({ frameData: new Uint8Array(pixels) });
    } catch (e) {
      console.error("get_preview_frame failed:", e);
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
}));
