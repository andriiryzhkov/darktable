// Client-side histogram and waveform computation from pixel data
// Supports both BGRA (raw SHM) and RGBA (decoded JPEG) layouts

export interface HistogramData {
  r: Uint32Array; // 256 bins
  g: Uint32Array;
  b: Uint32Array;
  max: number; // max bin value across all channels
}

export interface WaveformData {
  r: Uint8Array; // cols * rows density grid
  g: Uint8Array;
  b: Uint8Array;
  cols: number;
  rows: number;
  max: number;
}

const HIST_BINS = 256;

/**
 * Compute RGB histogram from pixel data.
 * @param isBGRA - true for BGRA layout [B,G,R,A], false for RGBA layout [R,G,B,A]
 */
export function computeHistogram(pixels: Uint8Array, isBGRA: boolean): HistogramData {
  const r = new Uint32Array(HIST_BINS);
  const g = new Uint32Array(HIST_BINS);
  const b = new Uint32Array(HIST_BINS);

  // Offsets: BGRA → r=2,g=1,b=0 | RGBA → r=0,g=1,b=2
  const rOff = isBGRA ? 2 : 0;
  const bOff = isBGRA ? 0 : 2;

  const len = pixels.length;
  for (let i = 0; i < len; i += 4) {
    r[pixels[i + rOff]]++;
    g[pixels[i + 1]]++;
    b[pixels[i + bOff]]++;
  }

  let max = 0;
  for (let i = 0; i < HIST_BINS; i++) {
    if (r[i] > max) max = r[i];
    if (g[i] > max) max = g[i];
    if (b[i] > max) max = b[i];
  }

  return { r, g, b, max };
}

const WAVE_COLS = 256;
const WAVE_ROWS = 256;

/**
 * Compute waveform from pixel data.
 * Fixed 256×256 internal grid — renderer scales to display size smoothly.
 * Y axis = brightness (row 0 = darkest, row 255 = brightest).
 * @param isBGRA - true for BGRA layout, false for RGBA layout
 */
export function computeWaveform(
  pixels: Uint8Array,
  width: number,
  height: number,
  isBGRA: boolean,
): WaveformData {
  const cols = WAVE_COLS;
  const rows = WAVE_ROWS;

  const rOff = isBGRA ? 2 : 0;
  const bOff = isBGRA ? 0 : 2;

  const total = cols * rows;
  const rAcc = new Uint32Array(total);
  const gAcc = new Uint32Array(total);
  const bAcc = new Uint32Array(total);

  const colScale = cols / width;
  const toneScale = (rows - 1) / 255;

  for (let y = 0; y < height; y++) {
    const rowOff = y * width * 4;
    for (let x = 0; x < width; x++) {
      const px = rowOff + x * 4;
      const col = Math.min(Math.floor(x * colScale), cols - 1);

      const rv = pixels[px + rOff];
      const gv = pixels[px + 1];
      const bv = pixels[px + bOff];

      rAcc[col * rows + Math.round(rv * toneScale)]++;
      gAcc[col * rows + Math.round(gv * toneScale)]++;
      bAcc[col * rows + Math.round(bv * toneScale)]++;
    }
  }

  // Per-column normalization: each column's max density → 255
  const r = new Uint8Array(total);
  const g = new Uint8Array(total);
  const b = new Uint8Array(total);
  let globalMax = 0;

  for (let col = 0; col < cols; col++) {
    const base = col * rows;

    // Find per-column max across all channels
    let colMax = 0;
    for (let row = 0; row < rows; row++) {
      const i = base + row;
      if (rAcc[i] > colMax) colMax = rAcc[i];
      if (gAcc[i] > colMax) colMax = gAcc[i];
      if (bAcc[i] > colMax) colMax = bAcc[i];
    }
    if (colMax > globalMax) globalMax = colMax;

    if (colMax > 0) {
      const logMax = Math.log1p(colMax);
      for (let row = 0; row < rows; row++) {
        const i = base + row;
        r[i] = Math.min(255, Math.round(Math.log1p(rAcc[i]) / logMax * 255));
        g[i] = Math.min(255, Math.round(Math.log1p(gAcc[i]) / logMax * 255));
        b[i] = Math.min(255, Math.round(Math.log1p(bAcc[i]) / logMax * 255));
      }
    }
  }

  return { r, g, b, cols, rows, max: globalMax };
}
