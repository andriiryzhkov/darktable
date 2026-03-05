interface BauhausLabelProps {
  label: string;
  value: string;
}

export default function BauhausLabel({ label, value }: BauhausLabelProps) {
  return (
    <div className="bauhaus-label">
      <span className="bauhaus-label-label">{label}:</span>
      <span className="bauhaus-label-value">{value}</span>
    </div>
  );
}
