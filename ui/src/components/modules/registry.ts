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
    component: lazy(() => import("./lib/HistoryModule")),
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
  {
    op: "mask_manager",
    name: "mask manager",
    component: lazy(() => import("./lib/MaskManagerModule")),
    views: VIEW_DARKROOM,
    container: PANEL_LEFT_CENTER,
    position: 10,
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
    position: 0,
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
//
// op names and groups match the C sources in src/iop/*.c
// Each module has its own file — implemented ones have full UI,
// stubs show "module controls not yet implemented".
// ---------------------------------------------------------------------------

export const IOP_MODULES: IopModuleDef[] = [
  // ── basic group ──────────────────────────────────────────────────────
  {
    op: "exposure",
    name: "exposure",
    component: lazy(() => import("./iop/ExposureModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_TECHNICAL,
    tags: ["brightness", "black point", "ev"],
  },
  {
    op: "crop",
    name: "crop",
    component: lazy(() => import("./iop/CropModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_TECHNICAL,
    tags: ["aspect ratio", "crop", "trim"],
  },
  {
    op: "flip",
    name: "orientation",
    component: lazy(() => import("./iop/OrientationModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_TECHNICAL,
    tags: ["rotate", "flip", "auto"],
  },
  {
    op: "demosaic",
    name: "demosaic",
    component: lazy(() => import("./iop/DemosaicModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_TECHNICAL,
    tags: ["bayer", "xtrans", "interpolation"],
  },
  {
    op: "rawprepare",
    name: "raw black/white point",
    component: lazy(() => import("./iop/RawprepareModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_TECHNICAL,
    tags: ["raw", "black point", "white point"],
  },
  {
    op: "highlights",
    name: "highlight reconstruction",
    component: lazy(() => import("./iop/HighlightsModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_TECHNICAL,
    tags: ["clipping", "blown", "recovery"],
  },
  {
    op: "colorreconstruction",
    name: "color reconstruction",
    component: lazy(() => import("./iop/ColorReconstructionModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_TECHNICAL,
    tags: ["blown highlights", "color recovery"],
  },
  {
    op: "negadoctor",
    name: "negadoctor",
    component: lazy(() => import("./iop/NegadoctorModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_TECHNICAL,
    tags: ["negative", "film scan", "inversion"],
  },
  {
    op: "enlargecanvas",
    name: "enlarge canvas",
    component: lazy(() => import("./iop/EnlargeCanvasModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_TECHNICAL,
    tags: ["canvas", "extend", "border"],
  },
  {
    op: "shadhi",
    name: "shadows and highlights",
    component: lazy(() => import("./iop/ShadhiModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_GRADING,
    tags: ["shadows", "highlights", "local contrast"],
  },
  {
    op: "toneequal",
    name: "tone equalizer",
    component: lazy(() => import("./iop/ToneEqualModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_GRADING,
    tags: ["tone", "equalizer", "dodging", "burning"],
  },
  {
    op: "basecurve",
    name: "base curve",
    component: lazy(() => import("./iop/BasecurveModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_TECHNICAL,
    tags: ["curve", "contrast"],
  },
  {
    op: "basicadj",
    name: "basic adjustments",
    component: lazy(() => import("./iop/BasicAdjModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_GRADING,
    tags: ["brightness", "contrast", "saturation", "deprecated"],
  },

  // ── tone group ───────────────────────────────────────────────────────
  {
    op: "sigmoid",
    name: "sigmoid",
    component: lazy(() => import("./iop/SigmoidModule")),
    defaultGroup: IOP_GROUP_TONE | IOP_GROUP_TECHNICAL,
    tags: ["tone mapping", "filmic", "display transform"],
  },
  {
    op: "filmicrgb",
    name: "filmic rgb",
    component: lazy(() => import("./iop/FilmicRgbModule")),
    defaultGroup: IOP_GROUP_TONE | IOP_GROUP_TECHNICAL,
    tags: ["tone mapping", "scene referred", "display transform"],
  },
  {
    op: "agx",
    name: "AgX",
    component: lazy(() => import("./iop/AgxModule")),
    defaultGroup: IOP_GROUP_TONE | IOP_GROUP_TECHNICAL,
    tags: ["tone mapping", "display transform"],
  },
  {
    op: "filmic",
    name: "filmic",
    component: lazy(() => import("./iop/FilmicModule")),
    defaultGroup: IOP_GROUP_TONE | IOP_GROUP_TECHNICAL,
    tags: ["tone mapping", "deprecated"],
  },
  {
    op: "tonecurve",
    name: "tone curve",
    component: lazy(() => import("./iop/TonecurveModule")),
    defaultGroup: IOP_GROUP_TONE | IOP_GROUP_GRADING,
    tags: ["curve", "contrast", "s-curve"],
  },
  {
    op: "rgbcurve",
    name: "rgb curve",
    component: lazy(() => import("./iop/RgbCurveModule")),
    defaultGroup: IOP_GROUP_TONE | IOP_GROUP_GRADING,
    tags: ["curve", "rgb", "channel"],
  },
  {
    op: "rgblevels",
    name: "rgb levels",
    component: lazy(() => import("./iop/RgbLevelsModule")),
    defaultGroup: IOP_GROUP_TONE | IOP_GROUP_GRADING,
    tags: ["levels", "rgb", "channel"],
  },
  {
    op: "levels",
    name: "levels",
    component: lazy(() => import("./iop/LevelsModule")),
    defaultGroup: IOP_GROUP_TONE | IOP_GROUP_GRADING,
    tags: ["levels", "deprecated"],
  },
  {
    op: "bilat",
    name: "local contrast",
    component: lazy(() => import("./iop/BilatModule")),
    defaultGroup: IOP_GROUP_TONE | IOP_GROUP_EFFECTS,
    tags: ["local contrast", "bilateral", "clarity"],
  },
  {
    op: "globaltonemap",
    name: "global tonemap",
    component: lazy(() => import("./iop/GlobalTonemapModule")),
    defaultGroup: IOP_GROUP_TONE | IOP_GROUP_GRADING,
    tags: ["tonemap", "deprecated"],
  },
  {
    op: "relight",
    name: "fill light",
    component: lazy(() => import("./iop/RelightModule")),
    defaultGroup: IOP_GROUP_TONE | IOP_GROUP_GRADING,
    tags: ["fill", "light", "deprecated"],
  },
  {
    op: "zonesystem",
    name: "zone system",
    component: lazy(() => import("./iop/ZoneSystemModule")),
    defaultGroup: IOP_GROUP_TONE | IOP_GROUP_GRADING,
    tags: ["zones", "ansel adams", "deprecated"],
  },

  // ── color group ──────────────────────────────────────────────────────
  {
    op: "temperature",
    name: "white balance",
    component: lazy(() => import("./iop/TemperatureModule")),
    defaultGroup: IOP_GROUP_BASIC | IOP_GROUP_GRADING,
    tags: ["color temperature", "tint", "wb"],
  },
  {
    op: "colorbalancergb",
    name: "color balance rgb",
    component: lazy(() => import("./iop/ColorBalanceRgbModule")),
    defaultGroup: IOP_GROUP_COLOR | IOP_GROUP_GRADING,
    tags: ["color grading", "lift gamma gain", "4-way"],
  },
  {
    op: "channelmixerrgb",
    name: "color calibration",
    component: lazy(() => import("./iop/ChannelMixerRgbModule")),
    defaultGroup: IOP_GROUP_COLOR,
    tags: ["color calibration", "white balance", "channel mixer"],
  },
  {
    op: "colorequal",
    name: "color equalizer",
    component: lazy(() => import("./iop/ColorEqualModule")),
    defaultGroup: IOP_GROUP_COLOR,
    tags: ["hue", "saturation", "brightness", "color grading"],
  },
  {
    op: "colorin",
    name: "input color profile",
    component: lazy(() => import("./iop/ColorInModule")),
    defaultGroup: IOP_GROUP_COLOR | IOP_GROUP_TECHNICAL,
    tags: ["icc", "profile", "input"],
  },
  {
    op: "colorout",
    name: "output color profile",
    component: lazy(() => import("./iop/ColorOutModule")),
    defaultGroup: IOP_GROUP_COLOR | IOP_GROUP_TECHNICAL,
    tags: ["icc", "profile", "output"],
  },
  {
    op: "colorzones",
    name: "color zones",
    component: lazy(() => import("./iop/ColorZonesModule")),
    defaultGroup: IOP_GROUP_COLOR | IOP_GROUP_GRADING,
    tags: ["hue", "saturation", "lightness", "zones"],
  },
  {
    op: "velvia",
    name: "velvia",
    component: lazy(() => import("./iop/VelviaModule")),
    defaultGroup: IOP_GROUP_COLOR | IOP_GROUP_GRADING,
    tags: ["saturation", "vibrance", "deprecated"],
  },
  {
    op: "vibrance",
    name: "vibrance",
    component: lazy(() => import("./iop/VibranceModule")),
    defaultGroup: IOP_GROUP_COLOR | IOP_GROUP_GRADING,
    tags: ["saturation", "vibrance", "deprecated"],
  },
  {
    op: "colorbalance",
    name: "color balance",
    component: lazy(() => import("./iop/ColorBalanceModule")),
    defaultGroup: IOP_GROUP_COLOR | IOP_GROUP_GRADING,
    tags: ["lift", "gamma", "gain", "deprecated"],
  },
  {
    op: "colorchecker",
    name: "color look up table",
    component: lazy(() => import("./iop/ColorCheckerModule")),
    defaultGroup: IOP_GROUP_COLOR | IOP_GROUP_TECHNICAL,
    tags: ["lut", "color checker", "calibration"],
  },
  {
    op: "colorcontrast",
    name: "color contrast",
    component: lazy(() => import("./iop/ColorContrastModule")),
    defaultGroup: IOP_GROUP_COLOR | IOP_GROUP_GRADING,
    tags: ["contrast", "color"],
  },
  {
    op: "colorcorrection",
    name: "color correction",
    component: lazy(() => import("./iop/ColorCorrectionModule")),
    defaultGroup: IOP_GROUP_COLOR | IOP_GROUP_GRADING,
    tags: ["shadows", "highlights", "tint"],
  },
  {
    op: "channelmixer",
    name: "channel mixer",
    component: lazy(() => import("./iop/ChannelMixerModule")),
    defaultGroup: IOP_GROUP_COLOR | IOP_GROUP_GRADING,
    tags: ["channel", "mixer", "deprecated"],
  },
  {
    op: "lut3d",
    name: "LUT 3D",
    component: lazy(() => import("./iop/Lut3dModule")),
    defaultGroup: IOP_GROUP_COLOR | IOP_GROUP_TECHNICAL,
    tags: ["3d lut", "cube", "color grading"],
  },
  {
    op: "monochrome",
    name: "monochrome",
    component: lazy(() => import("./iop/MonochromeModule")),
    defaultGroup: IOP_GROUP_COLOR | IOP_GROUP_EFFECTS,
    tags: ["black and white", "b&w", "desaturate"],
  },
  {
    op: "primaries",
    name: "rgb primaries",
    component: lazy(() => import("./iop/PrimariesModule")),
    defaultGroup: IOP_GROUP_COLOR | IOP_GROUP_GRADING,
    tags: ["primaries", "rgb", "gamut"],
  },
  {
    op: "profile_gamma",
    name: "unbreak input profile",
    component: lazy(() => import("./iop/ProfileGammaModule")),
    defaultGroup: IOP_GROUP_COLOR | IOP_GROUP_TECHNICAL,
    tags: ["gamma", "linear", "profile"],
  },

  // ── correct group ───────────────────────────────────────────────────
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
    op: "denoiseprofile",
    name: "denoise (profiled)",
    component: lazy(() => import("./iop/DenoiseModule")),
    defaultGroup: IOP_GROUP_CORRECT | IOP_GROUP_TECHNICAL,
    tags: ["noise reduction", "nr"],
  },
  {
    op: "diffuse",
    name: "diffuse or sharpen",
    component: lazy(() => import("./iop/DiffuseModule")),
    defaultGroup: IOP_GROUP_EFFECTS,
    tags: ["diffuse", "sharpen", "denoise", "dehaze", "blur"],
  },
  {
    op: "sharpen",
    name: "sharpen",
    component: lazy(() => import("./iop/SharpenModule")),
    defaultGroup: IOP_GROUP_CORRECT | IOP_GROUP_EFFECTS,
    tags: ["usm", "unsharp mask", "sharpening"],
  },
  {
    op: "cacorrectrgb",
    name: "chromatic aberrations",
    component: lazy(() => import("./iop/CaCorrectRgbModule")),
    defaultGroup: IOP_GROUP_CORRECT | IOP_GROUP_TECHNICAL,
    tags: ["ca", "chromatic aberration", "fringing"],
  },
  {
    op: "hazeremoval",
    name: "haze removal",
    component: lazy(() => import("./iop/HazeRemovalModule")),
    defaultGroup: IOP_GROUP_CORRECT | IOP_GROUP_TECHNICAL,
    tags: ["haze", "dehaze", "fog"],
  },
  {
    op: "hotpixels",
    name: "hot pixels",
    component: lazy(() => import("./iop/HotPixelsModule")),
    defaultGroup: IOP_GROUP_CORRECT | IOP_GROUP_TECHNICAL,
    tags: ["hot pixels", "dead pixels", "stuck"],
  },
  {
    op: "nlmeans",
    name: "astrophoto denoise",
    component: lazy(() => import("./iop/NlmeansModule")),
    defaultGroup: IOP_GROUP_CORRECT | IOP_GROUP_TECHNICAL,
    tags: ["denoise", "non-local means", "astrophotography"],
  },
  {
    op: "rawdenoise",
    name: "raw denoise",
    component: lazy(() => import("./iop/RawDenoiseModule")),
    defaultGroup: IOP_GROUP_CORRECT | IOP_GROUP_TECHNICAL,
    tags: ["denoise", "raw", "wavelet"],
  },
  {
    op: "defringe",
    name: "defringe",
    component: lazy(() => import("./iop/DefringeModule")),
    defaultGroup: IOP_GROUP_CORRECT | IOP_GROUP_TECHNICAL,
    tags: ["fringing", "purple", "deprecated"],
  },
  {
    op: "dither",
    name: "dither or posterize",
    component: lazy(() => import("./iop/DitherModule")),
    defaultGroup: IOP_GROUP_CORRECT | IOP_GROUP_TECHNICAL,
    tags: ["dither", "posterize", "banding"],
  },
  {
    op: "liquify",
    name: "liquify",
    component: lazy(() => import("./iop/LiquifyModule")),
    defaultGroup: IOP_GROUP_CORRECT | IOP_GROUP_EFFECTS,
    tags: ["warp", "distort", "liquify"],
  },
  {
    op: "retouch",
    name: "retouch",
    component: lazy(() => import("./iop/RetouchModule")),
    defaultGroup: IOP_GROUP_CORRECT | IOP_GROUP_EFFECTS,
    tags: ["clone", "heal", "spot removal", "retouch"],
  },
  {
    op: "spots",
    name: "spot removal",
    component: lazy(() => import("./iop/SpotsModule")),
    defaultGroup: IOP_GROUP_CORRECT | IOP_GROUP_EFFECTS,
    tags: ["clone", "heal", "spot", "deprecated"],
  },
  {
    op: "atrous",
    name: "contrast equalizer",
    component: lazy(() => import("./iop/AtrousModule")),
    defaultGroup: IOP_GROUP_CORRECT | IOP_GROUP_EFFECTS,
    tags: ["wavelets", "contrast", "sharpness", "denoise"],
  },
  {
    op: "scalepixels",
    name: "scale pixels",
    component: lazy(() => import("./iop/ScalePixelsModule")),
    defaultGroup: IOP_GROUP_CORRECT | IOP_GROUP_TECHNICAL,
    tags: ["pixel aspect ratio", "anamorphic"],
  },

  // ── effect group ─────────────────────────────────────────────────────
  {
    op: "graduatednd",
    name: "graduated density",
    component: lazy(() => import("./iop/GraduatedNdModule")),
    defaultGroup: IOP_GROUP_EFFECT | IOP_GROUP_GRADING,
    tags: ["graduated", "nd", "filter", "sky"],
  },
  {
    op: "vignette",
    name: "vignetting",
    component: lazy(() => import("./iop/VignetteModule")),
    defaultGroup: IOP_GROUP_EFFECT | IOP_GROUP_EFFECTS,
    tags: ["vignette", "light falloff"],
  },
  {
    op: "splittoning",
    name: "split-toning",
    component: lazy(() => import("./iop/SplitToningModule")),
    defaultGroup: IOP_GROUP_EFFECT | IOP_GROUP_GRADING,
    tags: ["split toning", "shadows", "highlights", "color"],
  },
  {
    op: "grain",
    name: "grain",
    component: lazy(() => import("./iop/GrainModule")),
    defaultGroup: IOP_GROUP_EFFECT | IOP_GROUP_EFFECTS,
    tags: ["film grain", "noise", "texture"],
  },
  {
    op: "soften",
    name: "soften",
    component: lazy(() => import("./iop/SoftenModule")),
    defaultGroup: IOP_GROUP_EFFECT | IOP_GROUP_EFFECTS,
    tags: ["glow", "orton", "soft focus"],
  },
  {
    op: "bloom",
    name: "bloom",
    component: lazy(() => import("./iop/BloomModule")),
    defaultGroup: IOP_GROUP_EFFECT | IOP_GROUP_EFFECTS,
    tags: ["bloom", "glow", "light"],
  },
  {
    op: "blurs",
    name: "blurs",
    component: lazy(() => import("./iop/BlursModule")),
    defaultGroup: IOP_GROUP_EFFECTS | IOP_GROUP_EFFECT,
    tags: ["blur", "lens blur", "motion blur", "gaussian"],
  },
  {
    op: "highpass",
    name: "highpass",
    component: lazy(() => import("./iop/HighpassModule")),
    defaultGroup: IOP_GROUP_EFFECT | IOP_GROUP_EFFECTS,
    tags: ["highpass", "filter", "detail"],
  },
  {
    op: "lowpass",
    name: "lowpass",
    component: lazy(() => import("./iop/LowpassModule")),
    defaultGroup: IOP_GROUP_EFFECT | IOP_GROUP_EFFECTS,
    tags: ["lowpass", "blur", "gaussian"],
  },
  {
    op: "colorize",
    name: "colorize",
    component: lazy(() => import("./iop/ColorizeModule")),
    defaultGroup: IOP_GROUP_EFFECT | IOP_GROUP_GRADING,
    tags: ["colorize", "tint", "sepia"],
  },
  {
    op: "colormapping",
    name: "color mapping",
    component: lazy(() => import("./iop/ColorMappingModule")),
    defaultGroup: IOP_GROUP_EFFECT | IOP_GROUP_EFFECTS,
    tags: ["color transfer", "style"],
  },
  {
    op: "borders",
    name: "framing",
    component: lazy(() => import("./iop/BordersModule")),
    defaultGroup: IOP_GROUP_EFFECT | IOP_GROUP_EFFECTS,
    tags: ["border", "frame", "matte"],
  },
  {
    op: "watermark",
    name: "watermark",
    component: lazy(() => import("./iop/WatermarkModule")),
    defaultGroup: IOP_GROUP_EFFECT | IOP_GROUP_EFFECTS,
    tags: ["watermark", "text", "svg", "overlay"],
  },
  {
    op: "overlay",
    name: "composite",
    component: lazy(() => import("./iop/OverlayModule")),
    defaultGroup: IOP_GROUP_EFFECT | IOP_GROUP_EFFECTS,
    tags: ["overlay", "composite", "blend"],
  },
  {
    op: "censorize",
    name: "censorize",
    component: lazy(() => import("./iop/CensorizeModule")),
    defaultGroup: IOP_GROUP_EFFECT | IOP_GROUP_EFFECTS,
    tags: ["censor", "pixelate", "blur", "privacy"],
  },
  {
    op: "lowlight",
    name: "lowlight vision",
    component: lazy(() => import("./iop/LowlightModule")),
    defaultGroup: IOP_GROUP_EFFECT | IOP_GROUP_EFFECTS,
    tags: ["night vision", "scotopic", "low light"],
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
  ).sort((a, b) => Math.abs(b.position) - Math.abs(a.position));
}

/** Get IOP modules matching a group bitmask */
export function getIopModulesByGroup(group: number): IopModuleDef[] {
  return IOP_MODULES.filter((m) => (m.defaultGroup & group) !== 0);
}
