interface BauhausTextInputProps {
  value?: string;
  placeholder?: string;
  onChange?: (value: string) => void;
}

export default function BauhausTextInput({
  value = "",
  placeholder,
  onChange,
}: BauhausTextInputProps) {
  return (
    <input
      type="text"
      className="dt-input"
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange?.(e.target.value)}
    />
  );
}
