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
  iop_order: number;
  params_size: number;
  description?: ModuleDescription;
}

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

export interface TemperatureParams {
  red: number;
  green: number;
  blue: number;
  various: number;
  preset: number;
  temperature_k?: number;
  tint?: number;
}

export interface FlipParams {
  orientation: number; // dt_image_orientation_t bitmask
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
  deflicker_computed_exposure?: number;
}

export interface SigmoidParams {
  middle_grey_contrast: number;
  contrast_skewness: number;
  color_processing: number; // 0 = per channel, 1 = RGB ratio
  hue_preservation: number;
  display_white_target: number;
  display_black_target: number;
  base_primaries: number; // 0=working profile, 1=Rec2020, 2=Display P3, 3=Adobe RGB, 4=sRGB
  red_inset: number;
  red_rotation: number;
  green_inset: number;
  green_rotation: number;
  blue_inset: number;
  blue_rotation: number;
  purity: number;
}

export interface ColorProfileEntry {
  type: number;
  name: string;
  filename: string;
}

export interface ColorinParams {
  type: number;
  filename: string;
  intent: number;
  normalize: number;
  type_work: number;
  filename_work: string;
  input_profile_name: string;
  work_profile_name: string;
  input_profiles: ColorProfileEntry[];
  work_profiles: ColorProfileEntry[];
}

export interface ColoroutParams {
  type: number;
  filename: string;
  intent: number;
  output_profile_name: string;
  output_profiles: ColorProfileEntry[];
}

export interface RawprepareParams {
  raw_black_level_separate: [number, number, number, number];
  raw_white_point: number;
  flat_field: number;
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export interface DemosaicParams {
  demosaicing_method: number;
  green_eq: number;
  median_thrs: number;
  color_smoothing: number;
  lmmse_refine: number;
  dual_thrs: number;
  cs_enabled: boolean;
  cs_radius: number;
  cs_thrs: number;
  cs_boost: number;
  cs_iter: number;
  cs_center: number;
  sensor_type: "bayer" | "xtrans" | "bayer4" | "mono";
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
