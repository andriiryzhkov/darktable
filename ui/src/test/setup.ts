import "@testing-library/jest-dom/vitest";

/**
 * Mock all window.* binding functions that the C webview layer
 * normally injects via webview_bind().  Each returns a resolved
 * promise with a sensible default so stores and components can
 * be tested without a live backend.
 */

const noop = () => Promise.resolve({});

// Catalog
window.ping = () => Promise.resolve({ status: "ok" });
window.catalogQuery = () => Promise.resolve({ images: [], total: 0 });
window.catalogGetThumbnail = () => Promise.resolve({ imgid: 0, mime: "image/jpeg", data: "" });
window.catalogGetThumbnails = () => Promise.resolve({ thumbnails: [] });
window.catalogGetCollectionValues = () => Promise.resolve({ values: [] });
window.catalogGetFilmrolls = () => Promise.resolve({ filmrolls: [] });
window.catalogGetTags = () => Promise.resolve({ tags: [] });

// Develop
window.developOpen = () => Promise.resolve({ session_id: "test-session", preview_width: 800, preview_height: 600 });
window.developClose = noop;
window.developSetParams = noop;
window.developCommitParams = noop;
window.developResetParams = noop;
window.developGetParams = (_sid: string, op: string) => Promise.resolve({ op, enabled: true, params: {} });
window.developRequestPreview = () => Promise.resolve({ status: "ok" });
window.developSamplePixels = () => Promise.resolve({ lab: [50, 0, 0], rgb: [128, 128, 128] });
window.developGetModules = () => Promise.resolve({ modules: [] });
window.developGetHistory = () => Promise.resolve({ items: [], history_end: 0 });
window.developSelectHistory = noop;
window.developCompressHistory = noop;
window.developTruncateHistory = noop;
window.developDeleteHistory = noop;
window.developListPresets = () => Promise.resolve({ presets: [] });
window.developApplyPreset = noop;
window.developStorePreset = noop;
window.developDeletePreset = noop;
window.developNewInstance = noop;
window.developDeleteInstance = noop;
window.developMoveInstance = noop;
window.developRenameInstance = noop;
window.developGetIntrospection = () => Promise.resolve({ op: "", fields: [] });

// Preview frame
window.getPreviewFrame = () => Promise.resolve({ data: "", width: 800, height: 600, front_buffer: 0, sequence: 1 });
window.getFramePort = () => Promise.resolve(0);

// File system
window.pickFolder = () => Promise.resolve(null);
window.listFolders = () => Promise.resolve([]);
window.listFiles = () => Promise.resolve([]);
window.getHomePath = () => Promise.resolve("/home/test");
window.importImages = () => Promise.resolve({ imported: 0, skipped: 0 });
window.copyAndImportImages = () => Promise.resolve({ imported: 0, skipped: 0 });
window.getFileThumbnail = () => Promise.resolve({ path: "", mime: "", data: "" });
window.getPlatformInfo = () => Promise.resolve({ os: "linux" as const });

// Config
window.configGet = (key: string) => Promise.resolve({ key, value: "" });
window.configSet = noop;

// Window management
window.windowStartDrag = () => Promise.resolve();
window.windowZoom = () => Promise.resolve();

// Event bridge
(window as Record<string, unknown>).__dt_event = () => {};
