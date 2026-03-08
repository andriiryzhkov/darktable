import type {
  CatalogQueryResult,
  ThumbnailResult,
  BatchThumbnailResult,
  SessionInfo,
  PreviewResult,
  PreviewFrameResult,
  PixelSampleResult,
  HistoryResult,
  ModuleInfo,
  FilmRoll,
  Tag,
  PresetListResult,
  IntrospectionResult,
} from "../types/protocol";
import type { CollectionRuleParam, PropertyValue } from "../types/collections";
import type { FolderEntry, FileEntry } from "../types/import";

// webview_bind() creates these as global async functions on window.
// Each returns a Promise that resolves when the C side calls webview_return().

declare global {
  interface Window {
    ping: () => Promise<unknown>;
    catalogQuery: (offset: number, limit: number, rules?: CollectionRuleParam[], sort?: string, sortOrder?: string) => Promise<CatalogQueryResult>;
    catalogGetThumbnail: (imgid: number) => Promise<ThumbnailResult>;
    catalogGetThumbnails: (imgids: number[], size?: number) => Promise<BatchThumbnailResult>;
    developOpen: (imgid: number, width: number, height: number) => Promise<SessionInfo>;
    developClose: (sessionId: string) => Promise<unknown>;
    developSetParams: (sessionId: string, op: string, params: Record<string, unknown>, previewOnly?: boolean) => Promise<unknown>;
    developCommitParams: (sessionId: string, op: string) => Promise<unknown>;
    developResetParams: (sessionId: string, op: string) => Promise<unknown>;
    developGetParams: (sessionId: string, op: string) => Promise<{ op: string; enabled: boolean; params: Record<string, unknown> }>;
    developRequestPreview: (sessionId: string) => Promise<PreviewResult>;
    developSamplePixels: (sessionId: string, x: number, y: number, w: number, h: number) => Promise<PixelSampleResult>;
    developGetModules: (sessionId: string) => Promise<{ modules: ModuleInfo[] }>;
    developGetHistory: (sessionId: string) => Promise<HistoryResult>;
    developSelectHistory: (sessionId: string, historyEnd: number) => Promise<unknown>;
    developCompressHistory: (sessionId: string) => Promise<unknown>;
    developTruncateHistory: (sessionId: string, historyEnd: number) => Promise<unknown>;
    developDeleteHistory: (sessionId: string) => Promise<unknown>;
    developListPresets: (sessionId: string, op: string) => Promise<PresetListResult>;
    developApplyPreset: (sessionId: string, op: string, name: string) => Promise<unknown>;
    developStorePreset: (sessionId: string, op: string, name: string, description?: string, filters?: string) => Promise<unknown>;
    developDeletePreset: (op: string, name: string) => Promise<unknown>;
    developNewInstance: (sessionId: string, op: string, instance?: number, copyParams?: boolean) => Promise<unknown>;
    developDeleteInstance: (sessionId: string, op: string, instance: number) => Promise<unknown>;
    developMoveInstance: (sessionId: string, op: string, instance: number, direction: "up" | "down") => Promise<unknown>;
    developRenameInstance: (sessionId: string, op: string, instance: number, name: string) => Promise<unknown>;
    developGetIntrospection: (sessionId: string, op: string) => Promise<IntrospectionResult>;
    getPreviewFrame: (sessionId: string, frontBuffer: number, format?: string) => Promise<PreviewFrameResult>;
    getFramePort: () => Promise<number>;
    pickFolder: () => Promise<string | null>;
    listFolders: (path: string) => Promise<FolderEntry[]>;
    listFiles: (path: string, recursive: boolean, ignoreNonRaw: boolean) => Promise<FileEntry[]>;
    getHomePath: () => Promise<string>;
    importImages: (paths: string[]) => Promise<{ imported: number; skipped: number }>;
    copyAndImportImages: (paths: string[]) => Promise<{ imported: number; skipped: number }>;
    getFileThumbnail: (path: string) => Promise<{ path: string; mime: string; data: string }>;
    getPlatformInfo: () => Promise<{ os: "macos" | "windows" | "linux" }>;
    catalogGetCollectionValues: (property: string, filter: string) => Promise<{ values: PropertyValue[] }>;
    catalogGetFilmrolls: () => Promise<{ filmrolls: FilmRoll[] }>;
    catalogGetTags: () => Promise<{ tags: Tag[] }>;
    configGet: (key: string) => Promise<{ key: string; value: string }>;
    configSet: (key: string, value: string) => Promise<unknown>;
    windowStartDrag: () => Promise<void>;
    windowZoom: () => Promise<void>;
  }
}

export type PlatformOS = "macos" | "windows" | "linux";

export const ping = () => window.ping();

export const catalogQuery = (offset: number, limit: number, rules?: CollectionRuleParam[], sort?: string, sortOrder?: string) =>
  window.catalogQuery(offset, limit, rules, sort, sortOrder);

export const catalogGetThumbnail = (imgid: number) =>
  window.catalogGetThumbnail(imgid);

export const catalogGetThumbnails = (imgids: number[], size?: number) =>
  window.catalogGetThumbnails(imgids, size);

export const developOpen = (imgid: number, width: number, height: number) =>
  window.developOpen(imgid, width, height);

export const developClose = (sessionId: string) =>
  window.developClose(sessionId);

export const developSetParams = (
  sessionId: string,
  op: string,
  params: Record<string, unknown>,
  previewOnly?: boolean,
) => window.developSetParams(sessionId, op, params, previewOnly);

export const developCommitParams = (sessionId: string, op: string) =>
  window.developCommitParams(sessionId, op);

export const developResetParams = (sessionId: string, op: string) =>
  window.developResetParams(sessionId, op);

export const developGetParams = (sessionId: string, op: string) =>
  window.developGetParams(sessionId, op);

export const developRequestPreview = (sessionId: string) =>
  window.developRequestPreview(sessionId);

export const developSamplePixels = (sessionId: string, x: number, y: number, w: number, h: number) =>
  window.developSamplePixels(sessionId, x, y, w, h);

export const developGetModules = (sessionId: string) =>
  window.developGetModules(sessionId);

export const developGetHistory = (sessionId: string) =>
  window.developGetHistory(sessionId);

export const developSelectHistory = (sessionId: string, historyEnd: number) =>
  window.developSelectHistory(sessionId, historyEnd);

export const developCompressHistory = (sessionId: string) =>
  window.developCompressHistory(sessionId);

export const developTruncateHistory = (sessionId: string, historyEnd: number) =>
  window.developTruncateHistory(sessionId, historyEnd);

export const developDeleteHistory = (sessionId: string) =>
  window.developDeleteHistory(sessionId);

export const developListPresets = (sessionId: string, op: string) =>
  window.developListPresets(sessionId, op);

export const developApplyPreset = (sessionId: string, op: string, name: string) =>
  window.developApplyPreset(sessionId, op, name);

export const developStorePreset = (sessionId: string, op: string, name: string, description?: string, filters?: import("../components/modules/StorePresetDialog").PresetFilterParams) =>
  window.developStorePreset(sessionId, op, name, description, filters ? JSON.stringify(filters) : undefined);

export const developDeletePreset = (op: string, name: string) =>
  window.developDeletePreset(op, name);

export const developNewInstance = (sessionId: string, op: string, instance?: number, copyParams?: boolean) =>
  window.developNewInstance(sessionId, op, instance, copyParams);

export const developDeleteInstance = (sessionId: string, op: string, instance: number) =>
  window.developDeleteInstance(sessionId, op, instance);

export const developMoveInstance = (sessionId: string, op: string, instance: number, direction: "up" | "down") =>
  window.developMoveInstance(sessionId, op, instance, direction);

export const developRenameInstance = (sessionId: string, op: string, instance: number, name: string) =>
  window.developRenameInstance(sessionId, op, instance, name);

export const developGetIntrospection = (sessionId: string, op: string) =>
  window.developGetIntrospection(sessionId, op);

export const getPreviewFrame = (sessionId: string, frontBuffer: number, format?: string) =>
  window.getPreviewFrame(sessionId, frontBuffer, format);

export const getFramePort = () => window.getFramePort();

export const pickFolder = () => window.pickFolder();

export const listFolders = (path: string) => window.listFolders(path);

export const listFiles = (path: string, recursive: boolean, ignoreNonRaw: boolean) =>
  window.listFiles(path, recursive, ignoreNonRaw);

export const getHomePath = () => window.getHomePath();

export const importImages = (paths: string[]) => window.importImages(paths);

export const copyAndImportImages = (paths: string[]) => window.copyAndImportImages(paths);

export const getFileThumbnail = (path: string) => window.getFileThumbnail(path);

export const getPlatformInfo = () => window.getPlatformInfo();

export const catalogGetCollectionValues = (property: string, filter: string) =>
  window.catalogGetCollectionValues(property, filter);

export const catalogGetFilmrolls = () => window.catalogGetFilmrolls();

export const catalogGetTags = () => window.catalogGetTags();

export const configGet = (key: string) => window.configGet(key);

export const configSet = (key: string, value: string) => window.configSet(key, value);

export const windowStartDrag = () => window.windowStartDrag();

export const windowZoom = () => window.windowZoom();
