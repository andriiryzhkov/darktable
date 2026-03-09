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
  status: string;
  session_id: string;
  pipeline_seq: number;
}

export interface ModuleDescription {
  main: string;
  purpose: string;
  input: string;
  process: string;
  output: string;
}

/** Blend parameters for a module */
export interface BlendParams {
  mask_mode: number;
  blend_mode: number;
  blend_parameter: number;
  opacity: number;
  mask_id: number;
  mask_combine: number;
  details: number;
  feathering_guide: number;
  feathering_radius: number;
  blur_radius: number;
  contrast: number;
  brightness: number;
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
  blend?: BlendParams;
}

/** IOP module flags from dt_iop_flags_t */
export const IOP_FLAGS = {
  SUPPORTS_BLENDING: 1 << 1,
  ONE_INSTANCE: 1 << 7,
  NO_MASKS: 1 << 10,
} as const;

/** Mask mode flags from dt_develop_mask_mode_t */
export const MASK_MODE = {
  DISABLED: 0,
  ENABLED: 1,       // uniform (no mask)
  DRAWN: 1 << 1,    // drawn mask
  PARAMETRIC: 1 << 2,
  RASTER: 1 << 3,
  DRAWN_PARAMETRIC: (1 << 1) | (1 << 2),
} as const;

/** Blend modes from dt_develop_blend_mode_t */
export const BLEND_MODE = {
  NORMAL2: 0x18,
  BOUNDED: 0x19,
  LIGHTEN: 0x02,
  DARKEN: 0x03,
  MULTIPLY: 0x04,
  AVERAGE: 0x05,
  ADD: 0x06,
  SUBTRACT: 0x07,
  DIFFERENCE2: 0x17,
  SCREEN: 0x09,
  OVERLAY: 0x0A,
  SOFTLIGHT: 0x0B,
  HARDLIGHT: 0x0C,
  VIVIDLIGHT: 0x0D,
  LINEARLIGHT: 0x0E,
  PINLIGHT: 0x0F,
  LIGHTNESS: 0x10,
  CHROMATICITY: 0x11,
  HUE: 0x12,
  COLOR: 0x13,
  COLORADJUST: 0x16,
  LAB_LIGHTNESS: 0x1A,
  LAB_COLOR: 0x1B,
  HSV_VALUE: 0x1C,
  HSV_COLOR: 0x1D,
  LAB_L: 0x1E,
  LAB_A: 0x1F,
  LAB_B: 0x20,
  RGB_R: 0x21,
  RGB_G: 0x22,
  RGB_B: 0x23,
  SUBTRACT_INVERSE: 0x25,
  DIVIDE: 0x26,
  DIVIDE_INVERSE: 0x27,
  GEOMETRIC_MEAN: 0x28,
  HARMONIC_MEAN: 0x29,
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

/** Mask type flags from dt_masks_type_t */
export const MASKS_TYPE = {
  NONE: 0,
  CIRCLE: 1 << 0,
  PATH: 1 << 1,
  GROUP: 1 << 2,
  CLONE: 1 << 3,
  GRADIENT: 1 << 4,
  ELLIPSE: 1 << 5,
  BRUSH: 1 << 6,
  NON_CLONE: 1 << 7,
} as const;

export interface MaskGroupChild {
  formid: number;
  state: number;
  opacity: number;
}

/** Point geometry for circle masks */
export interface MaskPointsCircle {
  center: [number, number];
  radius: number;
  border: number;
}

/** Point geometry for ellipse masks */
export interface MaskPointsEllipse {
  center: [number, number];
  radius: [number, number];
  rotation: number;
  border: number;
  flags: number; // 0=equidistant, 1=proportional
}

/** Single control point for path masks */
export interface MaskPointPath {
  corner: [number, number];
  ctrl1: [number, number];
  ctrl2: [number, number];
  border: [number, number];
  state: number;
}

/** Single control point for brush masks */
export interface MaskPointBrush {
  corner: [number, number];
  ctrl1: [number, number];
  ctrl2: [number, number];
  border: [number, number];
  density: number;
  hardness: number;
  state: number;
}

/** Point geometry for gradient masks */
export interface MaskPointsGradient {
  anchor: [number, number];
  rotation: number;
  compression: number;
  steepness: number;
  curvature: number;
  state: number; // 1=linear, 2=sigmoidal
}

export interface MaskForm {
  formid: number;
  name: string;
  type: number;
  type_name: string;
  is_clone: boolean;
  source?: [number, number];
  children?: MaskGroupChild[];
  points?: MaskPointsCircle | MaskPointsEllipse | MaskPointPath[] | MaskPointBrush[] | MaskPointsGradient;
}

export interface MaskUsage {
  mask_id: number;
  op: string;
  instance: number;
  module_name?: string;
}

export interface MaskListResult {
  forms: MaskForm[];
  usage: MaskUsage[];
}
