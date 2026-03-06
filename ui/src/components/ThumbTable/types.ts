/** Matches dt_thumbnail_overlay_t from darktable */
export enum OverlayMode {
  None = 0,
  HoverNormal = 1,
  HoverExtended = 2,
  AlwaysNormal = 3,
  AlwaysExtended = 4,
  Mixed = 5, // permanent normal, extended on hover
  HoverBlock = 6,
}

/** Matches dt_thumbtable_mode_t from darktable */
export enum ThumbTableMode {
  Filemanager = 1,
  Filmstrip = 2,
}
