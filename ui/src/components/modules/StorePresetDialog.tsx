import { useState } from "react";
import DialogOverlay from "../DialogOverlay";
import BauhausButton from "../controls/BauhausButton";
import BauhausInput from "../controls/BauhausInput";
import BauhausCheckbox from "../controls/BauhausCheckbox";

export interface PresetFilterParams {
  autoapply: boolean;
  filter: boolean;
  model: string;
  maker: string;
  lens: string;
  iso_min: number;
  iso_max: number;
  exposure_min: number;
  exposure_max: number;
  aperture_min: number;
  aperture_max: number;
  focal_length_min: number;
  focal_length_max: number;
  format: number; // bitmask: 1=raw, 2=non-raw, 4=hdr, 8=mono, 16=color
}

// From dt_gui_presets_exposure_value / dt_gui_presets_exposure_value_str
const EXPOSURE_VALUES: { label: string; value: number }[] = [
  { label: "0", value: 0 },
  { label: "1/8000", value: 1/8000 },
  { label: "1/4000", value: 1/4000 },
  { label: "1/2000", value: 1/2000 },
  { label: "1/1000", value: 1/1000 },
  { label: "1/500", value: 1/500 },
  { label: "1/250", value: 1/250 },
  { label: "1/125", value: 1/125 },
  { label: "1/60", value: 1/60 },
  { label: "1/30", value: 1/30 },
  { label: "1/15", value: 1/15 },
  { label: "1/8", value: 1/8 },
  { label: "1/4", value: 1/4 },
  { label: "1/2", value: 1/2 },
  { label: "1\"", value: 1 },
  { label: "2\"", value: 2 },
  { label: "4\"", value: 4 },
  { label: "8\"", value: 8 },
  { label: "15\"", value: 15 },
  { label: "30\"", value: 30 },
  { label: "60\"", value: 60 },
  { label: "+", value: Number.MAX_VALUE },
];

// From dt_gui_presets_aperture_value / dt_gui_presets_aperture_value_str
const APERTURE_VALUES: { label: string; value: number }[] = [
  { label: "f/0", value: 0 },
  { label: "f/0.95", value: 0.95 },
  { label: "f/1.0", value: 1.0 },
  { label: "f/1.2", value: 1.2 },
  { label: "f/1.4", value: 1.4 },
  { label: "f/1.8", value: 1.8 },
  { label: "f/2", value: 2.0 },
  { label: "f/2.4", value: 2.4 },
  { label: "f/2.8", value: 2.8 },
  { label: "f/4", value: 4.0 },
  { label: "f/5.6", value: 5.6 },
  { label: "f/8", value: 8.0 },
  { label: "f/11", value: 11.0 },
  { label: "f/16", value: 16.0 },
  { label: "f/22", value: 22.0 },
  { label: "f/32", value: 32.0 },
  { label: "f/45", value: 45.0 },
  { label: "f/64", value: 64.0 },
  { label: "f/90", value: 90.0 },
  { label: "f/128", value: 128.0 },
  { label: "f/+", value: Number.MAX_VALUE },
];

const EXPOSURE_LABELS = EXPOSURE_VALUES.map((e) => e.label);
const APERTURE_LABELS = APERTURE_VALUES.map((a) => a.label);

function exposureLabelToValue(label: string): number {
  return EXPOSURE_VALUES.find((e) => e.label === label)?.value ?? 0;
}

function apertureLabelToValue(label: string): number {
  return APERTURE_VALUES.find((a) => a.label === label)?.value ?? 0;
}

/** Find the closest label in a value list for a given numeric value. */
function closestLabel(values: { label: string; value: number }[], target: number): string {
  let best = values[0];
  let bestDiff = Math.abs(target - best.value);
  for (const v of values) {
    const diff = Math.abs(target - v.value);
    if (diff < bestDiff) {
      best = v;
      bestDiff = diff;
    }
  }
  return best.label;
}

export interface ImageDefaults {
  maker?: string;
  model?: string;
  lens?: string;
  iso?: number;
  exposure?: number;
  aperture?: number;
  focal_length?: number;
}

interface Props {
  moduleName: string;
  imageDefaults?: ImageDefaults;
  onConfirm: (name: string, description: string, filters?: PresetFilterParams) => void;
  onCancel: () => void;
}

export default function StorePresetDialog({ moduleName, imageDefaults, onConfirm, onCancel }: Props) {
  const img = imageDefaults;
  const [name, setName] = useState("new preset");
  const [description, setDescription] = useState("");
  const [autoapply, setAutoapply] = useState(false);
  const [filter, setFilter] = useState(false);
  const [model, setModel] = useState(img?.model || "%");
  const [maker, setMaker] = useState(img?.maker || "%");
  const [lens, setLens] = useState(img?.lens || "%");
  const [isoMin, setIsoMin] = useState(img?.iso ?? 0);
  const [isoMax, setIsoMax] = useState(img?.iso ?? 51200);
  const [exposureMinLabel, setExposureMinLabel] = useState(() =>
    img?.exposure ? closestLabel(EXPOSURE_VALUES, img.exposure) : "0"
  );
  const [exposureMaxLabel, setExposureMaxLabel] = useState(() =>
    img?.exposure ? closestLabel(EXPOSURE_VALUES, img.exposure) : "+"
  );
  const [apertureMinLabel, setApertureMinLabel] = useState(() =>
    img?.aperture ? closestLabel(APERTURE_VALUES, img.aperture) : "f/0"
  );
  const [apertureMaxLabel, setApertureMaxLabel] = useState(() =>
    img?.aperture ? closestLabel(APERTURE_VALUES, img.aperture) : "f/+"
  );
  const [focalMin, setFocalMin] = useState(img?.focal_length ?? 0);
  const [focalMax, setFocalMax] = useState(img?.focal_length ?? 1000);
  const [format, setFormat] = useState(0b11111);

  const showFilters = autoapply || filter;

  const hasRawOrNonRaw = (format & 3) !== 0; // bit 1 (raw) or bit 2 (non-raw)
  const hasColorGroup = (format & 28) !== 0;  // bit 4 (HDR) or bit 8 (mono) or bit 16 (color)
  const formatValid = !showFilters || (hasRawOrNonRaw && hasColorGroup);
  const nameValid = name.trim().length > 0;
  const okDisabled = !nameValid || !formatValid;

  const toggleFormatBit = (bit: number) => {
    setFormat((f) => f ^ bit);
  };

  const handleConfirm = () => {
    const trimmed = name.trim();
    if (!trimmed || !formatValid) return;
    const filters: PresetFilterParams | undefined = showFilters
      ? {
          autoapply,
          filter,
          model,
          maker,
          lens,
          iso_min: isoMin,
          iso_max: isoMax,
          exposure_min: exposureLabelToValue(exposureMinLabel),
          exposure_max: exposureLabelToValue(exposureMaxLabel),
          aperture_min: apertureLabelToValue(apertureMinLabel),
          aperture_max: apertureLabelToValue(apertureMaxLabel),
          focal_length_min: focalMin,
          focal_length_max: focalMax,
          format,
        }
      : undefined;
    onConfirm(trimmed, description.trim(), filters);
  };

  return (
    <DialogOverlay
      title={`store preset for module '${moduleName}'`}
      onClose={onCancel}
      zIndex={300}
    >
      <div className="store-preset-dialog">
        <BauhausInput value={name} onChange={setName} placeholder="name" />
        <BauhausInput value={description} onChange={setDescription} placeholder="description" />

        <BauhausCheckbox label="reset all module parameters to default values" />
        <BauhausCheckbox label="auto apply this preset to matching images" checked={autoapply} onChange={setAutoapply} />
        <BauhausCheckbox label="only show this preset for matching images" checked={filter} onChange={setFilter} />

        {showFilters && (
          <div className="store-preset-filters">
            <BauhausInput label="model" value={model} onChange={setModel} />
            <BauhausInput label="maker" value={maker} onChange={setMaker} />
            <BauhausInput label="lens" value={lens} onChange={setLens} />

            <div className="store-preset-range-row">
              <span className="store-preset-range-label">ISO</span>
              <BauhausInput value={String(isoMin)} onChange={(v) => setIsoMin(Number(v) || 0)} />
              <BauhausInput value={String(isoMax)} onChange={(v) => setIsoMax(Number(v) || 0)} placeholder="∞" />
            </div>
            <div className="store-preset-range-row">
              <span className="store-preset-range-label">exposure</span>
              <BauhausInput type="select" options={EXPOSURE_LABELS} value={exposureMinLabel} onChange={setExposureMinLabel} />
              <BauhausInput type="select" options={EXPOSURE_LABELS} value={exposureMaxLabel} onChange={setExposureMaxLabel} />
            </div>
            <div className="store-preset-range-row">
              <span className="store-preset-range-label">aperture</span>
              <BauhausInput type="select" options={APERTURE_LABELS} value={apertureMinLabel} onChange={setApertureMinLabel} />
              <BauhausInput type="select" options={APERTURE_LABELS} value={apertureMaxLabel} onChange={setApertureMaxLabel} />
            </div>
            <div className="store-preset-range-row">
              <span className="store-preset-range-label">focal length</span>
              <BauhausInput type="integer" value={focalMin} onChange={setFocalMin} min={0} max={1000} step={1} />
              <BauhausInput type="integer" value={focalMax} onChange={setFocalMax} min={0} max={1000} step={1} />
            </div>

            <div className="store-preset-format-grid">
              <span className="store-preset-format-label">format</span>
              <BauhausCheckbox label="non-raw" checked={(format & 2) !== 0} onChange={() => toggleFormatBit(2)} />
              <BauhausCheckbox label="HDR" checked={(format & 4) !== 0} onChange={() => toggleFormatBit(4)} disabled={!hasRawOrNonRaw} />
              <span />
              <span className="store-preset-format-and">and</span>
              <BauhausCheckbox label="monochrome" checked={(format & 8) !== 0} onChange={() => toggleFormatBit(8)} disabled={!hasRawOrNonRaw} />
              <span />
              <BauhausCheckbox label="raw" checked={(format & 1) !== 0} onChange={() => toggleFormatBit(1)} />
              <BauhausCheckbox label="color" checked={(format & 16) !== 0} onChange={() => toggleFormatBit(16)} disabled={!hasRawOrNonRaw} />
            </div>
          </div>
        )}

        <div className="store-preset-buttons">
          <BauhausButton label="export..." disabled />
          <BauhausButton label="delete" disabled />
          <BauhausButton label="cancel" onClick={onCancel} />
          <BauhausButton label="ok" onClick={handleConfirm} disabled={okDisabled} />
        </div>
      </div>
    </DialogOverlay>
  );
}
