import { ParentSize } from '@visx/responsive';
import { niceTicks } from '../lib/format';
import { showTip, hideTip, tooltipHeader, tooltipRow } from '../ui/Tooltip';

export interface ScatterPoint {
  x: number;
  y: number;
  r?: number;
  label?: string;
}

interface ScatterProps {
  points: ScatterPoint[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
  padding?: { t: number; r: number; b: number; l: number };
  regression?: boolean;
  xLine?: number | null;
  yLine?: number | null;
  colorBy?: (d: ScatterPoint) => string;
}

const DEFAULT_PADDING = { t: 10, r: 10, b: 26, l: 52 };

export function Scatter(props: ScatterProps) {
  return (
    <ParentSize>
      {({ width }) => <ScatterInner {...props} width={Math.max(200, width)} />}
    </ParentSize>
  );
}

function ScatterInner({
  points, height = 200,
  xLabel = 'x', yLabel = 'y',
  xFormat = (v) => v.toFixed(2), yFormat = (v) => v.toFixed(2),
  padding = DEFAULT_PADDING,
  regression = false, xLine = null, yLine = null,
  colorBy, width,
}: ScatterProps & { width: number }) {
  const w = width;
  const p = padding;
  const plotW = w - p.l - p.r;
  const plotH = height - p.t - p.b;

  if (!points.length) return <svg className="chart" width={w} height={height} />;

  const xs = points.map((d) => d.x);
  const ys = points.map((d) => d.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const dx = (xMax - xMin) * 0.05 || 1;
  const dy = (yMax - yMin) * 0.05 || 1;
  const X = (v: number) => p.l + (v - xMin + dx) / (xMax - xMin + 2 * dx) * plotW;
  const Y = (v: number) => p.t + (1 - (v - yMin + dy) / (yMax - yMin + 2 * dy)) * plotH;
  const xt = niceTicks(xMin, xMax, 5);
  const yt = niceTicks(yMin, yMax, 4);

  let reg: { b0: number; b1: number } | null = null;
  if (regression && points.length > 3) {
    const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
    const my = ys.reduce((a, b) => a + b, 0) / ys.length;
    let num = 0, den = 0;
    for (let i = 0; i < xs.length; i++) {
      num += (xs[i] - mx) * (ys[i] - my);
      den += (xs[i] - mx) ** 2;
    }
    if (den > 0) {
      const b1 = num / den;
      reg = { b0: my - b1 * mx, b1 };
    }
  }

  return (
    <svg className="chart" width={w} height={height}>
      {yt.map((v, k) => (
        <g key={k}>
          <line x1={p.l} x2={w - p.r} y1={Y(v)} y2={Y(v)}
            stroke="var(--rule)" strokeDasharray="2 3" />
          <text className="axis" x={p.l - 6} y={Y(v) + 3} textAnchor="end">
            {yFormat(v)}
          </text>
        </g>
      ))}
      {xt.map((v, k) => (
        <g key={k}>
          <line x1={X(v)} x2={X(v)} y1={p.t} y2={height - p.b}
            stroke="var(--rule)" strokeDasharray="2 3" />
          <text className="axis" x={X(v)} y={height - p.b + 12} textAnchor="middle">
            {xFormat(v)}
          </text>
        </g>
      ))}
      {xLine != null && (
        <line x1={X(xLine)} x2={X(xLine)} y1={p.t} y2={height - p.b}
          stroke="var(--warn)" strokeDasharray="3 3" />
      )}
      {yLine != null && (
        <line x1={p.l} x2={w - p.r} y1={Y(yLine)} y2={Y(yLine)}
          stroke="var(--warn)" strokeDasharray="3 3" />
      )}
      {points.map((d, i) => (
        <circle key={i} cx={X(d.x)} cy={Y(d.y)} r={d.r ?? 2.5}
          fill={colorBy ? colorBy(d) : 'var(--accent)'} opacity={0.75}
          onMouseMove={(e) => {
            let html = tooltipHeader(d.label ?? 'point');
            html += tooltipRow('var(--ink-2)', xLabel, xFormat(d.x));
            html += tooltipRow('var(--ink-2)', yLabel, yFormat(d.y));
            showTip(html, e);
          }}
          onMouseLeave={hideTip} />
      ))}
      {reg && (
        <line x1={X(xMin)} x2={X(xMax)}
          y1={Y(reg.b0 + reg.b1 * xMin)} y2={Y(reg.b0 + reg.b1 * xMax)}
          stroke="var(--warn)" strokeWidth={1} strokeDasharray="4 3" />
      )}
      <text className="axis" x={w - p.r} y={height - p.b + 12} textAnchor="end"
        style={{ fill: 'var(--muted)' }}>
        {xLabel}
      </text>
      <text className="axis" x={p.l} y={p.t + 10} style={{ fill: 'var(--muted)' }}>
        {yLabel}
      </text>
    </svg>
  );
}
