import { ParentSize } from '@visx/responsive';
import { showTip, hideTip, tooltipHeader, tooltipRow } from '../ui/Tooltip';

interface HeatmapProps {
  rows: string[];
  cols: string[];
  data: (number | null)[][];
  height?: number;
  fmt?: (v: number) => string;
  domain?: number | null;
  padding?: { t: number; r: number; b: number; l: number };
  cellPadding?: number;
  diverging?: boolean;
}

const DEFAULT_PADDING = { t: 14, r: 10, b: 22, l: 48 };

export function Heatmap(props: HeatmapProps) {
  return (
    <ParentSize>
      {({ width }) => <HeatmapInner {...props} width={Math.max(200, width)} />}
    </ParentSize>
  );
}

function HeatmapInner({
  rows, cols, data, height = 180,
  fmt = (v) => `${(v * 100).toFixed(1)}%`,
  domain = null, padding = DEFAULT_PADDING,
  cellPadding = 1, diverging = true, width,
}: HeatmapProps & { width: number }) {
  const w = width;
  const p = padding;
  const plotW = w - p.l - p.r;
  const plotH = height - p.t - p.b;
  const cw = plotW / cols.length;
  const ch = plotH / rows.length;

  const flat = data.flat().filter((v): v is number => v != null && isFinite(v));
  const absMax = domain ?? (flat.length ? Math.max(...flat.map(Math.abs)) : 1);

  const color = (v: number | null) => {
    if (v == null) return 'var(--rule)';
    const t = v / absMax;
    if (diverging) {
      if (t >= 0) {
        const a = Math.min(1, t);
        return `color-mix(in oklab, var(--pos) ${Math.round(a * 80 + 10)}%, #0b1620)`;
      }
      const a = Math.min(1, -t);
      return `color-mix(in oklab, var(--neg) ${Math.round(a * 80 + 10)}%, #0b1620)`;
    }
    const a = Math.min(1, Math.abs(t));
    return `color-mix(in oklab, var(--accent) ${Math.round(a * 80 + 10)}%, #0b1620)`;
  };

  return (
    <svg className="chart" width={w} height={height}>
      {rows.map((r, ri) => (
        <text key={ri} className="axis"
          x={p.l - 6} y={p.t + ri * ch + ch / 2 + 3} textAnchor="end">
          {r}
        </text>
      ))}
      {cols.map((c, ci) => (
        <text key={ci} className="axis"
          x={p.l + ci * cw + cw / 2} y={p.t + plotH + 12} textAnchor="middle">
          {c}
        </text>
      ))}
      {rows.map((r, ri) => cols.map((c, ci) => {
        const v = data[ri]?.[ci] ?? null;
        return (
          <g key={`${ri}-${ci}`}
            onMouseMove={(e) => {
              let html = tooltipHeader(`${r} · ${c}`);
              html += tooltipRow('var(--ink-2)', 'value', v == null ? '—' : fmt(v));
              showTip(html, e);
            }}
            onMouseLeave={hideTip}>
            <rect x={p.l + ci * cw + cellPadding / 2} y={p.t + ri * ch + cellPadding / 2}
              width={cw - cellPadding} height={ch - cellPadding} fill={color(v)} />
            {cw > 36 && v != null && (
              <text x={p.l + ci * cw + cw / 2} y={p.t + ri * ch + ch / 2 + 3}
                textAnchor="middle"
                style={{
                  fontSize: '9px',
                  fill: Math.abs(v / absMax) > 0.5 ? '#0b1017' : 'var(--ink-2)',
                  fontFamily: 'IBM Plex Mono',
                }}>
                {fmt(v)}
              </text>
            )}
          </g>
        );
      }))}
    </svg>
  );
}
