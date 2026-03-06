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
    getPreviewFrame: (sessionId: string, frontBuffer: number, format?: string) => Promise<PreviewFrameResult>;
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

export const getPreviewFrame = (sessionId: string, frontBuffer: number, format?: string) =>
  window.getPreviewFrame(sessionId, frontBuffer, format);

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
