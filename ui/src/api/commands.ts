import { invoke } from "@tauri-apps/api/core";
import type {
  CatalogQueryResult,
  ThumbnailResult,
  SessionInfo,
  PreviewResult,
} from "../types/protocol";

export const startServer = () => invoke<string>("start_server");

export const ping = () => invoke<Record<string, unknown>>("ping");

export const catalogQuery = (offset: number, limit: number) =>
  invoke<CatalogQueryResult>("catalog_query", { offset, limit });

export const catalogGetThumbnail = (imgid: number) =>
  invoke<ThumbnailResult>("catalog_get_thumbnail", { imgid });

export const developOpen = (imgid: number, width: number, height: number) =>
  invoke<SessionInfo>("develop_open", { imgid, width, height });

export const developClose = (sessionId: string) =>
  invoke<unknown>("develop_close", { sessionId });

export const developSetParams = (
  sessionId: string,
  op: string,
  params: Record<string, unknown>,
) => invoke<unknown>("develop_set_params", { sessionId, op, params });

export const developRequestPreview = (sessionId: string) =>
  invoke<PreviewResult>("develop_request_preview", { sessionId });

export const getPreviewFrame = (sessionId: string, frontBuffer: number) =>
  invoke<number[]>("get_preview_frame", { sessionId, frontBuffer });
