export interface ImageInfo {
  id: number;
  film_id: number;
  filename: string;
  datetime_taken: string;
  flags: number;
  width: number;
  height: number;
  aspect_ratio: number;
  exposure: number;
  aperture: number;
  iso: number;
  focal_length: number;
  folder: string;
  // Extended fields
  rating?: number;        // 0-5, 6 = rejected
  color_labels?: number;  // bitmask: bit0=red, bit1=yellow, bit2=green, bit3=blue, bit4=purple
  group_id?: number;
  altered?: boolean;
  local_copy?: boolean;
  maker?: string;
  model?: string;
  lens?: string;
  focus_distance?: number;
  exposure_bias?: number;
  longitude?: number;
  latitude?: number;
  altitude?: number;
  version?: number;
  max_version?: number;
  output_width?: number;
  output_height?: number;
  import_timestamp?: number;
  change_timestamp?: number;
  export_timestamp?: number;
  print_timestamp?: number;
  whitebalance?: string;
  flash?: string;
  exposure_program?: string;
  metering_mode?: string;
  crop?: number;
  orientation?: number;
  file_size?: number;
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

export interface BatchThumbnailItem {
  imgid: number;
  width?: number;
  height?: number;
  data?: string;
  error?: string;
}

export interface BatchThumbnailResult {
  thumbnails: BatchThumbnailItem[];
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

export interface ModuleDescription {
  main: string;
  purpose: string;
  input: string;
  process: string;
  output: string;
}

export interface ModuleInfo {
  op: string;
  name: string;
  enabled: boolean;
  instance: number;
  multi_name: string;
  flags: number;
  iop_order: number;
  params_size: number;
  description?: ModuleDescription;
}

/** IOP module flags from dt_iop_flags_t */
export const IOP_FLAGS = {
  ONE_INSTANCE: 1 << 7,
} as const;

export interface HistoryItem {
  num: number;
  history_index: number;
  op: string;
  name: string;
  enabled: boolean;
  mandatory: boolean;
}

export interface HistoryResult {
  history_end: number;
  items: HistoryItem[];
}

export interface ColorProfileEntry {
  type: number;
  name: string;
  filename: string;
}

export interface PixelSampleResult {
  mean_r: number;
  mean_g: number;
  mean_b: number;
}

export interface PreviewFrameResult {
  width: number;
  height: number;
  format: "jpeg" | "raw" | "raw_bgra"; // jpeg = JPEG compressed, raw/raw_bgra = BGRA8 pixels
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

export interface PresetInfo {
  name: string;
  description: string;
  writeprotect: boolean;
  active: boolean;
}

export interface PresetListResult {
  presets: PresetInfo[];
}

export interface IntrospectionEnumValue {
  name: string;
  value: number;
  description?: string;
}

export interface IntrospectionField {
  name: string;
  type: string;
  min?: number;
  max?: number;
  default?: number | boolean;
  values?: IntrospectionEnumValue[];
  count?: number;
  element?: IntrospectionField;
  fields?: IntrospectionField[];
  description?: string;
}

export interface IntrospectionResult {
  op: string;
  params_version: number;
  fields: IntrospectionField[];
}
