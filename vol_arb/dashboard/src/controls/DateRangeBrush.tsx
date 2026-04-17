import { useEffect, useMemo, useRef, useState } from 'react';
import { useReport } from '../data/ReportProvider';

interface DateRangeBrushProps {
  value: [number, number];
  onChange: (v: [number, number]) => void;
  minWidth?: number;
}

export function DateRangeBrush({ value, onChange, minWidth = 30 }: DateRangeBrushProps) {
  const report = useReport();
  const N = report.meta.n_bars;
  const eq = report.bars_nc.equity;

  const ref = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<'a' | 'b' | null>(null);

  const mini = useMemo(() => {
    const step = Math.max(1, Math.ceil(N / 160));
    const out: number[] = [];
    for (let i = 0; i < N; i += step) out.push(eq[i]);
    return out;
  }, [N, eq]);

  const mMin = Math.min(...mini);
  const mMax = Math.max(...mini);
  const span = mMax - mMin || 1;
  const y = (v: number) => 4 + (1 - (v - mMin) / span) * 18;

  useEffect(() => {
    if (!drag) return;
    function move(e: MouseEvent) {
      if (!ref.current) return;
      const r = ref.current.getBoundingClientRect();
      const xPx = Math.max(0, Math.min(r.width, e.clientX - r.left));
      const idx = Math.round((xPx / r.width) * (N - 1));
      if (drag === 'a') onChange([Math.min(idx, value[1] - minWidth), value[1]]);
      else onChange([value[0], Math.max(idx, value[0] + minWidth)]);
    }
    function up() { setDrag(null); }
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
    return () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
    };
  }, [drag, value, N, minWidth, onChange]);

  const path = `M ${mini
    .map((v, i) => `${(i / (mini.length - 1) * 100).toFixed(2)} ${y(v).toFixed(1)}`)
    .join(' L ')}`;
  const a$ = (value[0] / (N - 1)) * 100;
  const b$ = (value[1] / (N - 1)) * 100;

  return (
    <div className="daterange" ref={ref}>
      <svg viewBox="0 0 100 26" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0 }}>
        <path d={path} fill="none" stroke="var(--muted)" strokeWidth={0.4} vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="sel" style={{ left: `${a$}%`, width: `${b$ - a$}%` }} />
      <div className="handle"
        style={{ left: `calc(${a$}% - 3px)` }}
        onMouseDown={(e) => { e.preventDefault(); setDrag('a'); }} />
      <div className="handle"
        style={{ left: `calc(${b$}% - 3px)` }}
        onMouseDown={(e) => { e.preventDefault(); setDrag('b'); }} />
    </div>
  );
}
