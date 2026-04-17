interface SegControlProps<T extends string> {
  options: readonly T[];
  value: T;
  onChange: (v: T) => void;
  labels?: Partial<Record<T, string>>;
}

export function SegControl<T extends string>({ options, value, onChange, labels }: SegControlProps<T>) {
  return (
    <span className="seg">
      {options.map((opt) => (
        <button key={opt} className={value === opt ? 'on' : ''} onClick={() => onChange(opt)}>
          {labels?.[opt] ?? opt}
        </button>
      ))}
    </span>
  );
}
