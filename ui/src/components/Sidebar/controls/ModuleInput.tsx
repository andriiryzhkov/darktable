import { useState, useCallback, useEffect, useRef, type ChangeEvent, type FocusEvent } from "react";

interface ModuleInputBaseProps {
  label: string;
  placeholder?: string;
}

interface ModuleInputTextProps extends ModuleInputBaseProps {
  type?: "text";
  value?: string;
  onChange?: (value: string) => void;
  min?: never;
  max?: never;
  step?: never;
  options?: never;
}

interface ModuleInputNumberProps extends ModuleInputBaseProps {
  type: "integer";
  value?: number;
  onChange?: (value: number) => void;
  min: number;
  max: number;
  step: number;
  options?: never;
}

interface ModuleInputSelectProps extends ModuleInputBaseProps {
  type: "select";
  options: string[];
  value?: string;
  onChange?: (value: string) => void;
  min?: never;
  max?: never;
  step?: never;
}

type ModuleInputProps = ModuleInputTextProps | ModuleInputNumberProps | ModuleInputSelectProps;

export default function ModuleInput(props: ModuleInputProps) {
  const { label, placeholder } = props;

  if (props.type === "integer") {
    return (
      <IntegerInput
        label={label}
        placeholder={placeholder}
        defaultValue={props.value}
        onChange={props.onChange}
        min={props.min}
        max={props.max}
        step={props.step}
      />
    );
  }

  if (props.type === "select") {
    return (
      <SelectInput
        label={label}
        options={props.options}
        defaultValue={props.value}
        onChange={props.onChange}
      />
    );
  }

  return (
    <TextInput
      label={label}
      placeholder={placeholder}
      value={props.value}
      onChange={props.onChange}
    />
  );
}

function TextInput({
  label,
  value: controlledValue,
  placeholder,
  onChange,
}: {
  label: string;
  value?: string;
  placeholder?: string;
  onChange?: (value: string) => void;
}) {
  const [internal, setInternal] = useState("");
  const value = controlledValue ?? internal;

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const next = e.target.value;
      setInternal(next);
      onChange?.(next);
    },
    [onChange],
  );

  return (
    <div className="bauhaus-input">
      <span className="bauhaus-input-label">{label}</span>
      <input
        type="text"
        className="bauhaus-input-field"
        value={value}
        placeholder={placeholder}
        onChange={handleChange}
      />
    </div>
  );
}

function IntegerInput({
  label,
  defaultValue,
  placeholder,
  onChange,
  min,
  max,
  step,
}: {
  label: string;
  defaultValue?: number;
  placeholder?: string;
  onChange?: (value: number) => void;
  min: number;
  max: number;
  step: number;
}) {
  const clamp = useCallback(
    (v: number) => Math.min(max, Math.max(min, v)),
    [min, max],
  );

  const [value, setValue] = useState(() => clamp(defaultValue ?? min));
  const [draft, setDraft] = useState(String(value));
  const [editing, setEditing] = useState(false);

  const apply = useCallback(
    (next: number) => {
      setValue(next);
      setDraft(String(next));
      onChange?.(next);
    },
    [onChange],
  );

  const decrement = useCallback(() => {
    apply(clamp(value - step));
  }, [value, step, clamp, apply]);

  const increment = useCallback(() => {
    apply(clamp(value + step));
  }, [value, step, clamp, apply]);

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      setDraft(e.target.value);
    },
    [],
  );

  const handleFocus = useCallback(() => {
    setEditing(true);
    setDraft(String(value));
  }, [value]);

  const handleBlur = useCallback(
    (e: FocusEvent<HTMLInputElement>) => {
      setEditing(false);
      const parsed = parseInt(e.target.value, 10);
      if (!isNaN(parsed)) {
        apply(clamp(parsed));
      } else {
        setDraft(String(value));
      }
    },
    [value, clamp, apply],
  );

  return (
    <div className="bauhaus-input">
      <span className="bauhaus-input-label">{label}</span>
      <button
        className="bauhaus-input-step"
        onClick={decrement}
        disabled={value <= min}
      >
        <svg viewBox="0 0 8 2" xmlns="http://www.w3.org/2000/svg">
          <path d="M0 1h8" stroke="currentColor" strokeWidth="1.5" />
        </svg>
      </button>
      <input
        type="text"
        className="bauhaus-input-field bauhaus-input-field-number"
        value={editing ? draft : value}
        placeholder={placeholder}
        onChange={handleChange}
        onFocus={handleFocus}
        onBlur={handleBlur}
      />
      <button
        className="bauhaus-input-step"
        onClick={increment}
        disabled={value >= max}
      >
        <svg viewBox="0 0 8 8" xmlns="http://www.w3.org/2000/svg">
          <path d="M0 4h8M4 0v8" stroke="currentColor" strokeWidth="1.5" />
        </svg>
      </button>
    </div>
  );
}

function SelectInput({
  label,
  options,
  defaultValue,
  onChange,
}: {
  label: string;
  options: string[];
  defaultValue?: string;
  onChange?: (value: string) => void;
}) {
  const [value, setValue] = useState(defaultValue ?? options[0] ?? "");
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleKey);
    };
  }, [open]);

  const handleSelect = useCallback(
    (option: string) => {
      setValue(option);
      setOpen(false);
      onChange?.(option);
    },
    [onChange],
  );

  return (
    <div className="bauhaus-input bauhaus-input-select" ref={ref}>
      <span className="bauhaus-input-label">{label}</span>
      <div
        className="bauhaus-input-field bauhaus-input-field-select"
        onClick={() => setOpen(!open)}
      >
        <span className="bauhaus-input-select-value">{value}</span>
        <svg
          className="bauhaus-input-select-arrow"
          viewBox="0 0 10 6"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path d="M0 0l5 6 5-6z" fill="currentColor" />
        </svg>
        {open && (
          <div className="bauhaus-input-select-popup">
            {options.map((option) => (
              <div
                key={option}
                className="bauhaus-input-select-option"
                data-selected={option === value}
                onClick={() => handleSelect(option)}
              >
                {option}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
