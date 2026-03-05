// Default module group presets matching darktable's built-in presets
// See src/libs/modulegroups.c init_presets()

export interface PresetGroup {
  id: string;
  label: string;
  icon: string; // lucide icon name
  modules: string[]; // iop op names
}

export interface ModuleGroupPreset {
  name: string;
  hasQuickAccess: boolean;
  groups: PresetGroup[];
}

export const MODULE_GROUP_PRESETS: ModuleGroupPreset[] = [
  {
    name: "modules: all",
    hasQuickAccess: true,
    groups: [
      {
        id: "base",
        label: "base",
        icon: "Circle",
        modules: [
          "basecurve", "crop", "ashift", "colisa", "colorreconstruct",
          "demosaic", "exposure", "flip", "highlights", "negadoctor",
          "rawprepare", "shadhi", "temperature", "toneequal",
        ],
      },
      {
        id: "tone",
        label: "tone",
        icon: "Sun",
        modules: [
          "agx", "bilat", "filmicrgb", "levels", "rgbcurve",
          "rgblevels", "sigmoid", "tonecurve",
        ],
      },
      {
        id: "color",
        label: "color",
        icon: "Palette",
        modules: [
          "channelmixerrgb", "colorbalancergb", "colorchecker",
          "colorcontrast", "colorcorrection", "colorin", "colorout",
          "colorzones", "colorequal", "lut3d", "monochrome",
          "profile_gamma", "primaries", "velvia",
        ],
      },
      {
        id: "correct",
        label: "correct",
        icon: "Wrench",
        modules: [
          "atrous", "bilateral", "cacorrect", "cacorrectrgb",
          "denoiseprofile", "dither", "hazeremoval", "hotpixels",
          "lens", "liquify", "nlmeans", "rawdenoise", "retouch",
          "sharpen",
        ],
      },
      {
        id: "effect",
        label: "effect",
        icon: "Sparkles",
        modules: [
          "bloom", "borders", "colorize", "colormapping",
          "enlargecanvas", "graduatednd", "grain", "highpass",
          "lowlight", "lowpass", "overlay", "soften", "splittoning",
          "vignette", "watermark", "censorize", "blurs", "diffuse",
        ],
      },
    ],
  },
  {
    name: "search only",
    hasQuickAccess: false,
    groups: [],
  },
  {
    name: "workflow: beginner",
    hasQuickAccess: true,
    groups: [
      {
        id: "base",
        label: "base",
        icon: "Circle",
        modules: [
          "ashift", "sigmoid", "basecurve", "crop", "denoiseprofile",
          "exposure", "flip", "lens", "temperature",
        ],
      },
      {
        id: "grading",
        label: "grading",
        icon: "SwatchBook",
        modules: [
          "channelmixerrgb", "colorequal", "graduatednd",
          "rgbcurve", "rgblevels", "splittoning",
        ],
      },
      {
        id: "effects",
        label: "effects",
        icon: "Sparkles",
        modules: [
          "borders", "monochrome", "retouch", "sharpen",
          "vignette", "watermark",
        ],
      },
    ],
  },
  {
    name: "workflow: display-referred",
    hasQuickAccess: false,
    groups: [
      {
        id: "base",
        label: "base",
        icon: "Circle",
        modules: [
          "basecurve", "toneequal", "crop", "ashift", "flip",
          "exposure", "temperature", "rgbcurve", "rgblevels",
          "bilat", "shadhi", "highlights",
        ],
      },
      {
        id: "color",
        label: "color",
        icon: "Palette",
        modules: [
          "channelmixerrgb", "colorbalancergb", "colorcorrection",
          "colorzones", "monochrome", "velvia",
        ],
      },
      {
        id: "correct",
        label: "correct",
        icon: "Wrench",
        modules: [
          "cacorrect", "cacorrectrgb", "denoiseprofile",
          "hazeremoval", "hotpixels", "lens", "retouch", "liquify",
          "sharpen", "nlmeans",
        ],
      },
      {
        id: "effect",
        label: "effect",
        icon: "Sparkles",
        modules: [
          "borders", "enlargecanvas", "colorize", "graduatednd",
          "grain", "overlay", "splittoning", "vignette", "watermark",
          "censorize",
        ],
      },
    ],
  },
  {
    name: "workflow: scene-referred",
    hasQuickAccess: true,
    groups: [
      {
        id: "base",
        label: "base",
        icon: "Circle",
        modules: [
          "filmicrgb", "sigmoid", "agx", "toneequal", "crop",
          "ashift", "flip", "exposure", "temperature", "bilat",
          "highlights",
        ],
      },
      {
        id: "color",
        label: "color",
        icon: "Palette",
        modules: [
          "channelmixerrgb", "colorbalancergb", "colorequal",
          "primaries",
        ],
      },
      {
        id: "correct",
        label: "correct",
        icon: "Wrench",
        modules: [
          "cacorrect", "cacorrectrgb", "denoiseprofile",
          "hazeremoval", "hotpixels", "lens", "retouch", "liquify",
          "sharpen", "nlmeans",
        ],
      },
      {
        id: "effect",
        label: "effect",
        icon: "Sparkles",
        modules: [
          "atrous", "borders", "enlargecanvas", "graduatednd",
          "grain", "overlay", "vignette", "watermark", "censorize",
          "blurs", "diffuse",
        ],
      },
    ],
  },
];

export const DEFAULT_PRESET_NAME = "workflow: scene-referred";
