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

export interface ExposureParams {
  mode: number;
  exposure: number;
  black: number;
  compensate_exposure_bias: boolean;
}
