interface ModuleTextInputProps {
  value?: string;
  placeholder?: string;
  onChange?: (value: string) => void;
}

export default function ModuleTextInput({
  value = "",
  placeholder,
  onChange,
}: ModuleTextInputProps) {
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
