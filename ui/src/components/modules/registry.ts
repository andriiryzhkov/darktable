import { lazy, type ComponentType } from "react";

// ---------------------------------------------------------------------------
// View flags — matches dt_view_type_flags_t (src/views/view.h)
// ---------------------------------------------------------------------------
export const VIEW_LIGHTTABLE = 1 << 0;
export const VIEW_DARKROOM = 1 << 1;
export const VIEW_TETHERING = 1 << 2;
export const VIEW_MAP = 1 << 3;
export const VIEW_SLIDESHOW = 1 << 4;
export const VIEW_PRINT = 1 << 5;

// ---------------------------------------------------------------------------
// Panel containers — matches dt_ui_container_t (src/gui/gtk.h)
// ---------------------------------------------------------------------------
export const PANEL_LEFT_TOP = 0;
export const PANEL_LEFT_CENTER = 1;
export const PANEL_LEFT_BOTTOM = 2;
export const PANEL_RIGHT_TOP = 3;
export const PANEL_RIGHT_CENTER = 4;
export const PANEL_RIGHT_BOTTOM = 5;
export const PANEL_TOP_LEFT = 6;
export const PANEL_TOP_CENTER = 7;
export const PANEL_TOP_RIGHT = 8;
export const PANEL_BOTTOM = 15;

// ---------------------------------------------------------------------------
// IOP module groups — matches dt_iop_group_t (src/develop/imageop.h)
// ---------------------------------------------------------------------------
export const IOP_GROUP_BASIC = 1 << 0;
export const IOP_GROUP_TONE = 1 << 1;
export const IOP_GROUP_COLOR = 1 << 2;
export const IOP_GROUP_CORRECT = 1 << 3;
export const IOP_GROUP_EFFECT = 1 << 4;
export const IOP_GROUP_TECHNICAL = 1 << 5;
export const IOP_GROUP_GRADING = 1 << 6;
export const IOP_GROUP_EFFECTS = 1 << 7;

// ---------------------------------------------------------------------------
// Lib module definition — mirrors dt_lib_module_t API
// ---------------------------------------------------------------------------
export interface LibModuleDef {
  op: string; // matches plugin_name
  name: string;
  component: ComponentType;
  views: number; // bitmask of VIEW_*
  container: number; // PANEL_*
  position: number; // order within container; negative flips sidebar
  expandable?: boolean; // default true
}

// ---------------------------------------------------------------------------
// IOP module definition — mirrors dt_iop_module_so_t API
// ---------------------------------------------------------------------------
export interface IopModuleDef {
  op: string;
  name: string;
  component: ComponentType;
  defaultGroup: number; // bitmask of IOP_GROUP_*
  tags?: string[];
}

// ---------------------------------------------------------------------------
// IOP module group tabs — mirrors dt_lib_modulegroups_group_t
// ---------------------------------------------------------------------------
export interface IopGroupDef {
  id: string;
  name: string;
  icon: string; // icon identifier
  modules: string[]; // list of op names in this group
}

// ---------------------------------------------------------------------------
// Lib modules — defaults match GTK darktable's views()/container()/position()
//
// Config key format (same as GTK DT darktablerc):
//   plugins/{view}/{layout}/{op}_visible
//   plugins/{view}/{layout}/{op}_position
// ---------------------------------------------------------------------------
export const LIB_MODULES: LibModuleDef[] = [
  // -- Darkroom left sidebar --
  {
    op: "navigation",
    name: "navigation",
    component: lazy(() => import("./lib/NavigationModule")),
    views: VIEW_DARKROOM,
    container: PANEL_LEFT_TOP,
    position: 1001,
    expandable: false,
  },
  {
    op: "metadata_view",
    name: "image information",
    component: lazy(() => import("./lib/ImageInfoModule")),
    views: VIEW_LIGHTTABLE | VIEW_DARKROOM | VIEW_TETHERING | VIEW_MAP,
    container: PANEL_LEFT_CENTER,
    position: 299,
  },
  {
    op: "history",
    name: "history",
    component: lazy(() => import("./lib/DarkroomHistoryModule")),
    views: VIEW_DARKROOM,
    container: PANEL_LEFT_CENTER,
    position: 900,
  },
  {
    op: "snapshots",
    name: "snapshots",
    component: lazy(() => import("./lib/SnapshotsModule")),
    views: VIEW_DARKROOM,
    container: PANEL_LEFT_CENTER,
    position: 1000,
  },

  // -- Lighttable left sidebar --
  {
    op: "import",
    name: "import",
    component: lazy(() => import("./lib/ImportModule")),
    views: VIEW_LIGHTTABLE,
    container: PANEL_LEFT_CENTER,
    position: 999,
  },
  {
    op: "collect",
    name: "collections",
    component: lazy(() => import("./lib/CollectionsModule")),
    views: VIEW_LIGHTTABLE | VIEW_MAP | VIEW_PRINT,
    container: PANEL_LEFT_CENTER,
    position: 400,
  },
  {
    op: "filtering",
    name: "collection filters",
    component: lazy(() => import("./lib/CollectionFiltersModule")),
    views: VIEW_LIGHTTABLE | VIEW_MAP | VIEW_PRINT,
    container: PANEL_LEFT_CENTER,
    position: 350,
  },
  {
    op: "scripts",
    name: "scripts",
    component: lazy(() => import("./lib/ScriptsModule")),
    views: VIEW_LIGHTTABLE,
    container: PANEL_LEFT_CENTER,
    position: 599,
  },

  // -- Lighttable right sidebar --
  {
    op: "select",
    name: "selection",
    component: lazy(() => import("./lib/SelectionModule")),
    views: VIEW_LIGHTTABLE,
    container: PANEL_RIGHT_CENTER,
    position: 800,
  },
  {
    op: "image",
    name: "actions on selection",
    component: lazy(() => import("./lib/ActionsModule")),
    views: VIEW_LIGHTTABLE,
    container: PANEL_RIGHT_CENTER,
    position: 700,
  },
  {
    op: "tagging",
    name: "tagging",
    component: lazy(() => import("./lib/TaggingModule")),
    views: VIEW_LIGHTTABLE | VIEW_DARKROOM | VIEW_MAP | VIEW_TETHERING,
    container: PANEL_RIGHT_CENTER,
    position: 500,
  },
  {
    op: "styles",
    name: "styles",
    component: lazy(() => import("./lib/StylesModule")),
    views: VIEW_LIGHTTABLE,
    container: PANEL_RIGHT_CENTER,
    position: 599,
  },
  {
    op: "metadata",
    name: "edit metadata",
    component: lazy(() => import("./lib/EditMetadataModule")),
    views: VIEW_LIGHTTABLE,
    container: PANEL_RIGHT_CENTER,
    position: 510,
  },
  {
    op: "copy_history",
    name: "history stack",
    component: lazy(() => import("./lib/HistoryStackModule")),
    views: VIEW_LIGHTTABLE,
    container: PANEL_RIGHT_CENTER,
    position: 600,
  },
  {
    op: "geotagging",
    name: "geotagging",
    component: lazy(() => import("./lib/GeotaggingModule")),
    views: VIEW_LIGHTTABLE | VIEW_MAP,
    container: PANEL_RIGHT_CENTER,
    position: 450,
  },
  {
    op: "export",
    name: "export",
    component: lazy(() => import("./lib/ExportModule")),
    views: VIEW_LIGHTTABLE | VIEW_DARKROOM,
    container: PANEL_RIGHT_CENTER,
    position: 0,
  },
];

// ---------------------------------------------------------------------------
// IOP (processing) modules
// ---------------------------------------------------------------------------
export const IOP_MODULES: IopModuleDef[] = [
  {
    op: "sigmoid",
    name: "sigmoid",
    component: lazy(() => import("./iop/SigmoidModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_TECHNICAL,
    tags: ["tone mapping", "filmic", "display transform"],
  },
  {
    op: "crop",
    name: "crop",
    component: lazy(() => import("./iop/CropModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_TECHNICAL,
    tags: ["aspect ratio", "crop", "trim"],
  },
  {
    op: "exposure",
    name: "exposure",
    component: lazy(() => import("./iop/ExposureModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_TECHNICAL,
    tags: ["brightness", "black point", "ev"],
  },
  {
    op: "orientation",
    name: "orientation",
    component: lazy(() => import("./iop/OrientationModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_TECHNICAL,
    tags: ["rotate", "flip", "auto"],
  },
  {
    op: "ashift",
    name: "rotate and perspective",
    component: lazy(() => import("./iop/AshiftModule")),
    defaultGroup: IOP_GROUP_CORRECT | IOP_GROUP_TECHNICAL,
    tags: ["keystone", "perspective", "rotation"],
  },
  {
    op: "lens",
    name: "lens correction",
    component: lazy(() => import("./iop/LensModule")),
    defaultGroup: IOP_GROUP_CORRECT | IOP_GROUP_TECHNICAL,
    tags: ["distortion", "vignetting", "ca", "chromatic"],
  },
  {
    op: "denoise",
    name: "denoise (profiled)",
    component: lazy(() => import("./iop/DenoiseModule")),
    defaultGroup: IOP_GROUP_CORRECT | IOP_GROUP_TECHNICAL,
    tags: ["noise reduction", "nr"],
  },
  {
    op: "temperature",
    name: "white balance",
    component: lazy(() => import("./iop/TemperatureModule")),
    defaultGroup: IOP_GROUP_COLOR | IOP_GROUP_TECHNICAL,
    tags: ["color temperature", "tint", "wb"],
  },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Get lib modules for a specific view and container, sorted by position */
export function getLibModules(
  view: number,
  container: number,
): LibModuleDef[] {
  return LIB_MODULES.filter(
    (m) => (m.views & view) !== 0 && m.container === container,
  ).sort((a, b) => Math.abs(a.position) - Math.abs(b.position));
}

/** Get IOP modules matching a group bitmask */
export function getIopModulesByGroup(group: number): IopModuleDef[] {
  return IOP_MODULES.filter((m) => (m.defaultGroup & group) !== 0);
}
