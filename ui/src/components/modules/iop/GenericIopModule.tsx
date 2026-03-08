import { useState, useEffect } from "react";
import { useDevelopStore } from "../../../stores/developStore";
import { useThrottledParam } from "../../../hooks/useThrottledParam";
import BauhausSlider from "../../controls/BauhausSlider";
import BauhausCombo from "../../controls/BauhausCombo";
import BauhausCheckbox from "../../controls/BauhausCheckbox";
import type { IntrospectionField } from "../../../types/protocol";

function humanize(name: string): string {
  return name.replace(/_/g, " ");
}

function isSliderType(type: string): boolean {
  return type === "float" || type === "double" || type === "int" || type === "uint" ||
         type === "short" || type === "ushort" || type === "int8" || type === "uint8";
}

function clampedMinMax(field: IntrospectionField): { min: number; max: number } {
  let min = field.min ?? 0;
  let max = field.max ?? 1;
  // Introspection defaults use G_MAXFLOAT/G_MINFLOAT for unconstrained fields.
  // Clamp to reasonable UI range.
  if (min < -1e6) min = -100;
  if (max > 1e6) max = 100;
  if (min >= max) { min = 0; max = 1; }
  return { min, max };
}

function stepForType(type: string, min: number, max: number): number {
  if (type === "int" || type === "uint" || type === "short" || type === "ushort" ||
      type === "int8" || type === "uint8") return 1;
  const range = max - min;
  if (range <= 2) return 0.001;
  if (range <= 20) return 0.01;
  if (range <= 200) return 0.1;
  return 1;
}

interface FieldControlProps {
  field: IntrospectionField;
  value: unknown;
  onApply: (field: string, v: unknown) => void;
  onSet: (field: string, v: unknown) => void;
  onRelease?: () => void;
}

function FieldControl({ field, value, onApply, onSet, onRelease }: FieldControlProps) {
  const [local, setLocal] = useState<number>(typeof value === "number" ? value : 0);

  useEffect(() => {
    if (typeof value === "number") setLocal(value);
  }, [value]);

  if (field.type === "bool") {
    return (
      <BauhausCheckbox
        label={humanize(field.name)}
        checked={!!value}
        onChange={(v) => onSet(field.name, v)}
      />
    );
  }

  if (field.type === "enum" && field.values) {
    const options = field.values.map((v) => v.description || humanize(v.name));
    const currentIdx = field.values.findIndex((v) => v.value === value);
    const currentLabel = currentIdx >= 0 ? options[currentIdx] : String(value);
    return (
      <BauhausCombo
        label={humanize(field.name)}
        options={options}
        value={currentLabel}
        onChange={(label) => {
          const idx = options.indexOf(label);
          if (idx >= 0) onSet(field.name, field.values![idx].value);
        }}
      />
    );
  }

  if (isSliderType(field.type)) {
    const { min, max } = clampedMinMax(field);
    const step = stepForType(field.type, min, max);
    const isInt = field.type !== "float" && field.type !== "double";
    const defaultVal = typeof field.default === "number" ? field.default : min;
    return (
      <BauhausSlider
        label={humanize(field.name)}
        value={local}
        min={min}
        max={max}
        step={step}
        defaultValue={defaultVal}
        format={(v) => isInt ? String(Math.round(v)) : v.toFixed(step < 0.01 ? 3 : step < 0.1 ? 2 : 1)}
        onChange={(v) => { setLocal(v); onApply(field.name, isInt ? Math.round(v) : v); }}
        onRelease={onRelease}
      />
    );
  }

  // Arrays and structs — skip for now
  return null;
}

export default function GenericIopModule({ op }: { op: string }) {
  const fetchIntrospection = useDevelopStore((s) => s.fetchIntrospection);
  const fetchGenericParams = useDevelopStore((s) => s.fetchGenericParams);
  const setModuleParam = useDevelopStore((s) => s.setModuleParam);
  const schema = useDevelopStore((s) => s.introspectionSchemas[op]);
  const params = useDevelopStore((s) => s.genericParams[op]);
  const { apply: throttledApply, release: throttledRelease } = useThrottledParam(op);

  useEffect(() => {
    fetchIntrospection(op);
    fetchGenericParams(op);
  }, [op, fetchIntrospection, fetchGenericParams]);

  if (!schema || !params) {
    return (
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        loading...
      </p>
    );
  }

  const fields = schema.fields.filter(
    (f) => f.type !== "opaque" && f.type !== "struct" && f.type !== "array" && f.name
  );

  if (fields.length === 0) {
    return (
      <p className="text-xs" style={{ color: "var(--disabled-fg-color)" }}>
        no adjustable parameters
      </p>
    );
  }

  return (
    <>
      {fields.map((field) => (
        <FieldControl
          key={field.name}
          field={field}
          value={params[field.name]}
          onApply={throttledApply}
          onSet={(name, v) => setModuleParam(op, { [name]: v })}
          onRelease={throttledRelease}
        />
      ))}
    </>
  );
}
