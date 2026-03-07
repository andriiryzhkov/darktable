import { describe, it, expect, vi, beforeEach } from "vitest";
import { useDevelopStore, getEnabledOps } from "../stores/developStore";
import type { HistoryItem } from "../types/protocol";

beforeEach(() => {
  // Reset store to initial state between tests
  useDevelopStore.setState({
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
    loading: false,
    previewError: null,
    focusModuleOp: null,
    zoom: "fit",
    panX: 0.5,
    panY: 0.5,
  });
});

describe("getEnabledOps", () => {
  it("returns empty set for empty history", () => {
    expect(getEnabledOps([])).toEqual(new Set());
  });

  it("returns enabled ops from history", () => {
    const items: HistoryItem[] = [
      { num: 1, op: "exposure", enabled: true, params: {} },
      { num: 2, op: "temperature", enabled: true, params: {} },
      { num: 3, op: "sigmoid", enabled: false, params: {} },
    ];
    const ops = getEnabledOps(items);
    expect(ops.has("exposure")).toBe(true);
    expect(ops.has("temperature")).toBe(true);
    expect(ops.has("sigmoid")).toBe(false);
  });

  it("uses last history entry for each op", () => {
    const items: HistoryItem[] = [
      { num: 1, op: "exposure", enabled: true, params: {} },
      { num: 2, op: "exposure", enabled: false, params: {} },
    ];
    const ops = getEnabledOps(items);
    expect(ops.has("exposure")).toBe(false);
  });
});

describe("useDevelopStore", () => {
  it("has correct initial state", () => {
    const state = useDevelopStore.getState();
    expect(state.sessionId).toBeNull();
    expect(state.imgid).toBeNull();
    expect(state.loading).toBe(false);
    expect(state.zoom).toBe("fit");
    expect(state.panX).toBe(0.5);
    expect(state.panY).toBe(0.5);
  });

  it("setZoom updates zoom and resets pan", () => {
    useDevelopStore.getState().setPan(0.3, 0.7);
    useDevelopStore.getState().setZoom("100");
    const state = useDevelopStore.getState();
    expect(state.zoom).toBe("100");
    expect(state.panX).toBe(0.5);
    expect(state.panY).toBe(0.5);
  });

  it("setPan updates pan coordinates", () => {
    useDevelopStore.getState().setPan(0.2, 0.8);
    const state = useDevelopStore.getState();
    expect(state.panX).toBe(0.2);
    expect(state.panY).toBe(0.8);
  });

  it("focusModule sets focusModuleOp", () => {
    useDevelopStore.getState().focusModule("exposure");
    expect(useDevelopStore.getState().focusModuleOp).toBe("exposure");
  });

  it("openSession sets sessionId and fetches metadata", async () => {
    window.developOpen = vi.fn().mockResolvedValue({
      session_id: "sess-42",
      preview_width: 1024,
      preview_height: 768,
    });
    window.developGetModules = vi.fn().mockResolvedValue({ modules: [] });
    window.developGetHistory = vi.fn().mockResolvedValue({ items: [], history_end: 0 });
    window.developRequestPreview = vi.fn().mockResolvedValue({ status: "ok" });

    await useDevelopStore.getState().openSession(1);

    const state = useDevelopStore.getState();
    expect(state.sessionId).toBe("sess-42");
    expect(state.imgid).toBe(1);
    expect(state.loading).toBe(false);
    expect(window.developOpen).toHaveBeenCalledWith(1, expect.any(Number), expect.any(Number));
  });

  it("closeSession clears state", async () => {
    useDevelopStore.setState({ sessionId: "sess-1", imgid: 5 });
    window.developClose = vi.fn().mockResolvedValue({});

    await useDevelopStore.getState().closeSession();

    const state = useDevelopStore.getState();
    expect(state.sessionId).toBeNull();
    expect(state.imgid).toBeNull();
    expect(state.modules).toEqual([]);
    expect(window.developClose).toHaveBeenCalledWith("sess-1");
  });

  it("samplePixels returns null without session", async () => {
    const result = await useDevelopStore.getState().samplePixels(0, 0, 10, 10);
    expect(result).toBeNull();
  });

  it("samplePixels calls binding with session", async () => {
    useDevelopStore.setState({ sessionId: "sess-1" });
    window.developSamplePixels = vi.fn().mockResolvedValue({ lab: [50, 0, 0], rgb: [128, 128, 128] });

    const result = await useDevelopStore.getState().samplePixels(10, 20, 5, 5);
    expect(result).toEqual({ lab: [50, 0, 0], rgb: [128, 128, 128] });
    expect(window.developSamplePixels).toHaveBeenCalledWith("sess-1", 10, 20, 5, 5);
  });
});
