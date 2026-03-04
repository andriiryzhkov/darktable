export interface ImageInfo {
  id: number;
  film_id: number;
  filename: string;
  datetime_taken: number;
  flags: number;
  width: number;
  height: number;
  aspect_ratio: number;
  exposure: number;
  aperture: number;
  iso: number;
  focal_length: number;
  folder: string;
  // Extended fields (may be absent from older server responses)
  rating?: number;        // 0-5, 6 = rejected
  color_labels?: number;  // bitmask: bit0=red, bit1=yellow, bit2=green, bit3=blue, bit4=purple
  group_id?: number;
  altered?: boolean;
  local_copy?: boolean;
}

export interface CatalogQueryResult {
  images: ImageInfo[];
  total: number;
  offset: number;
  limit: number;
}

export interface ThumbnailResult {
  imgid: number;
  width: number;
  height: number;
  format: string;
  encoding: string;
  data: string;
}

export interface SessionInfo {
  session_id: string;
  imgid: number;
  preview_width: number;
  preview_height: number;
  shm_names: string[];
}

export interface PreviewResult {
  session_id: string;
  width: number;
  height: number;
  sequence: number;
  shm_name: string;
  front_buffer: number;
}

export interface ModuleInfo {
  op: string;
  name: string;
  enabled: boolean;
  instance: number;
  iop_order: number;
  params_size: number;
}

export interface HistoryItem {
  num: number;
  op: string;
  name: string;
  enabled: boolean;
  mandatory: boolean;
}

export interface HistoryResult {
  history_end: number;
  items: HistoryItem[];
}

export interface ExposureParams {
  mode: number;
  exposure: number;
  black: number;
  compensate_exposure_bias: boolean;
  compensate_hilite_pres: boolean;
  deflicker_percentile: number;
  deflicker_target_level: number;
  exposure_bias_ev: number;
  highlight_bias_ev: number;
}

export interface PreviewFrameResult {
  width: number;
  height: number;
  format: "jpeg" | "raw"; // jpeg = JPEG compressed, raw = BGRA8 pixels
  data: string; // base64-encoded
}

export interface FilmRoll {
  id: number;
  folder: string;
  access_timestamp: number;
  image_count: number;
}

export interface Tag {
  id: number;
  name: string;
  flags: number;
}
