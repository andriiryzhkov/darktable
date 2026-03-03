export type CollectionMode = "and" | "or" | "and_not";

export type CollectionProperty =
  // files
  | "film_roll"
  | "folder"
  | "filename"
  // metadata
  | "tag"
  | "rating"
  | "color_label"
  | "title"
  | "description"
  | "creator"
  // times
  | "capture_date"
  | "import_time"
  | "change_time"
  // capture details
  | "camera"
  | "lens"
  | "aperture"
  | "exposure"
  | "exposure_bias"
  | "focal_length"
  | "iso"
  | "aspect_ratio"
  | "white_balance"
  | "flash"
  | "exposure_program"
  | "metering_mode"
  // darktable
  | "group"
  | "history";

export const PROPERTY_LABELS: Record<CollectionProperty, string> = {
  film_roll: "film roll",
  folder: "folder",
  filename: "filename",
  tag: "tag",
  rating: "rating",
  color_label: "color label",
  title: "title",
  description: "description",
  creator: "creator",
  capture_date: "capture date",
  import_time: "import time",
  change_time: "modification time",
  camera: "camera",
  lens: "lens",
  aperture: "aperture",
  exposure: "exposure",
  exposure_bias: "exposure bias",
  focal_length: "focal length",
  iso: "ISO",
  aspect_ratio: "aspect ratio",
  white_balance: "white balance",
  flash: "flash",
  exposure_program: "exposure program",
  metering_mode: "metering mode",
  group: "group",
  history: "history",
};

export interface PropertyCategory {
  label: string;
  properties: CollectionProperty[];
}

export const PROPERTY_CATEGORIES: PropertyCategory[] = [
  { label: "files", properties: ["film_roll", "folder", "filename"] },
  {
    label: "metadata",
    properties: ["tag", "rating", "color_label", "title", "description", "creator"],
  },
  { label: "times", properties: ["capture_date", "import_time", "change_time"] },
  {
    label: "capture details",
    properties: [
      "camera", "lens", "aperture", "exposure", "exposure_bias",
      "focal_length", "iso", "aspect_ratio", "white_balance",
      "flash", "exposure_program", "metering_mode",
    ],
  },
  { label: "darktable", properties: ["group", "history"] },
];

export const MODE_LABELS: Record<CollectionMode, string> = {
  and: "AND",
  or: "OR",
  and_not: "AND NOT",
};

export interface PropertyValue {
  id: number | string;
  label: string;
  count: number;
}

export interface CollectionRule {
  id: string;
  mode: CollectionMode;
  property: CollectionProperty;
  text: string;
  selectedValue: string | null;
}

export interface CollectionRuleParam {
  mode: CollectionMode;
  property: string;
  text: string;
}
