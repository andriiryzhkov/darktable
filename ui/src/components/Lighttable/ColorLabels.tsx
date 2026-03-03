const LABEL_BITS = [
  { bit: 0, var: "--colorlabel-red" },
  { bit: 1, var: "--colorlabel-yellow" },
  { bit: 2, var: "--colorlabel-green" },
  { bit: 3, var: "--colorlabel-blue" },
  { bit: 4, var: "--colorlabel-purple" },
];

interface ColorLabelsProps {
  labels: number;
  dotSize?: number;
}

export default function ColorLabels({
  labels,
  dotSize = 9,
}: ColorLabelsProps) {
  if (labels === 0) return null;

  return (
    <div className="flex items-center gap-0.5">
      {LABEL_BITS.map(
        (l) =>
          (labels & (1 << l.bit)) !== 0 && (
            <div
              key={l.bit}
              className="rounded-full"
              style={{
                width: dotSize,
                height: dotSize,
                backgroundColor: `var(${l.var})`,
              }}
            />
          ),
      )}
    </div>
  );
}
