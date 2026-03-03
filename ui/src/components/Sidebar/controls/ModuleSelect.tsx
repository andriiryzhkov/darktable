interface ModuleSelectProps {
  options: string[];
  value?: string;
  onChange?: (value: string) => void;
}

export default function ModuleSelect({
  options,
  value,
  onChange,
}: ModuleSelectProps) {
  return (
    <select
      className="dt-select"
      value={value ?? options[0]}
      onChange={(e) => onChange?.(e.target.value)}
    >
      {options.map((opt) => (
        <option key={opt} value={opt}>
          {opt}
        </option>
      ))}
    </select>
  );
}
