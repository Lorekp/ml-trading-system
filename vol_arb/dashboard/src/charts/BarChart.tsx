import { ParentSize } from '@visx/responsive';
import { niceTicks } from '../lib/format';
import { showTip, hideTip, tooltipHeader, tooltipRow } from '../ui/Tooltip';

export interface Bar {
  label: string;
  v: number;
  se?: number;
  n?: number;
  faded?: boolean;
}

interface BarChartProps {
  bars: Bar[];
  height?: number;
  yFormat?: (v: number) => string;
  padding?: { t: number; r: number; b: number; l: number };
  errorBars?: boolean;
  baseline?: number;
  labels?: boolean;
  colorMap?: (b: Bar, i: number) => string;
  tooltipLabel?: string;
}

const DEFAULT_PADDING = { t: 10, r: 10, b: 26, l: 48 };

export function BarChart(props: BarChartProps) {
  return (
    <ParentSize>
      {({ width }) => <BarChartInner {...props} width={Math.max(200, width)} />}
    </ParentSize>
  );
}

function BarChartInner({
  bars, height = 180, yFormat = (v) => v.toFixed(2),
  padding = DEFAULT_PADDING, errorBars = false, baseline = 0,
  labels = true, colorMap, tooltipLabel = 'bar', width,
}: BarChartProps & { width: number }) {
  const w = width;
  const h = height;
  const p = padding;
  const plotW = w - p.l - p.r;
  const plotH = h - p.t - p.b;

  if (!bars.length) return <svg className="chart" width={w} height={h} />;

  const ymaxP = Math.max(...bars.map((b) => b.v + (b.se ?? 0)), baseline);
  const yminP = Math.min(...bars.map((b) => b.v - (b.se ?? 0)), baseline);
  const pad = (ymaxP - yminP) * 0.1 || 1;
  const yMin = yminP - pad;
  const yMax = ymaxP + pad;

  const y = (v: number) => p.t + (1 - (v - yMin) / (yMax - yMin)) * plotH;
  const bw = plotW / bars.length;
  const yt = niceTicks(yMin, yMax, 4);

  return (
    <svg className="chart" width={w} height={h}>
      {yt.map((v, k) => (
        <g key={k}>
          <line x1={p.l} x2={w - p.r} y1={y(v)} y2={y(v)}
            stroke="var(--rule)" strokeDasharray={v === baseline ? '' : '2 3'} />
          <text className="axis" x={p.l - 6} y={y(v) + 3} textAnchor="end">
            {yFormat(v)}
          </text>
        </g>
      ))}
      {bars.map((b, i) => {
        const x = p.l + i * bw + bw * 0.12;
        const bwidth = bw * 0.76;
        const yv = y(b.v);
        const y0 = y(baseline);
        const color = colorMap ? colorMap(b, i) : (b.v >= baseline ? 'var(--pos)' : 'var(--neg)');
        const top = Math.min(yv, y0);
        const hgt = Math.abs(yv - y0);
        return (
          <g key={i}
            onMouseMove={(e) => {
              let html = tooltipHeader(`${tooltipLabel} ${b.label || i}`);
              html += tooltipRow('var(--ink-2)', 'value', yFormat(b.v));
              if (b.se != null) html += tooltipRow('var(--muted)', '±1 se', yFormat(b.se));
              if (b.n != null) html += tooltipRow('var(--muted)', 'n', String(b.n));
              showTip(html, e);
            }}
            onMouseLeave={hideTip}>
            <rect x={x} y={top} width={bwidth} height={Math.max(1, hgt)}
              fill={color} opacity={b.faded ? 0.35 : 0.85} />
            {errorBars && b.se != null && (
              <line x1={x + bwidth / 2} x2={x + bwidth / 2}
                y1={y(b.v - b.se)} y2={y(b.v + b.se)}
                stroke="var(--ink)" strokeWidth={1} />
            )}
            {labels && (
              <text className="axis" x={x + bwidth / 2} y={h - p.b + 12} textAnchor="middle">
                {b.label}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
