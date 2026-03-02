import type {
  CatalogQueryResult,
  ThumbnailResult,
  SessionInfo,
  PreviewResult,
  PreviewFrameResult,
} from "../types/protocol";

// webview_bind() creates these as global async functions on window.
// Each returns a Promise that resolves when the C side calls webview_return().

declare global {
  interface Window {
    ping: () => Promise<unknown>;
    catalogQuery: (offset: number, limit: number) => Promise<CatalogQueryResult>;
    catalogGetThumbnail: (imgid: number) => Promise<ThumbnailResult>;
    developOpen: (imgid: number, width: number, height: number) => Promise<SessionInfo>;
    developClose: (sessionId: string) => Promise<unknown>;
    developSetParams: (sessionId: string, op: string, params: Record<string, unknown>) => Promise<unknown>;
    developRequestPreview: (sessionId: string) => Promise<PreviewResult>;
    getPreviewFrame: (sessionId: string, frontBuffer: number) => Promise<PreviewFrameResult>;
  }
}

export const ping = () => window.ping();

export const catalogQuery = (offset: number, limit: number) =>
  window.catalogQuery(offset, limit);

export const catalogGetThumbnail = (imgid: number) =>
  window.catalogGetThumbnail(imgid);

export const developOpen = (imgid: number, width: number, height: number) =>
  window.developOpen(imgid, width, height);

export const developClose = (sessionId: string) =>
  window.developClose(sessionId);

export const developSetParams = (
  sessionId: string,
  op: string,
  params: Record<string, unknown>,
) => window.developSetParams(sessionId, op, params);

export const developRequestPreview = (sessionId: string) =>
  window.developRequestPreview(sessionId);

export const getPreviewFrame = (sessionId: string, frontBuffer: number) =>
  window.getPreviewFrame(sessionId, frontBuffer);
