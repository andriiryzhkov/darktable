interface BauhausSelectProps {
  options: string[];
  value?: string;
  onChange?: (value: string) => void;
}

export default function BauhausSelect({
  options,
  value,
  onChange,
}: BauhausSelectProps) {
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
