import { ParentSize } from '@visx/responsive';
import { niceTicks } from '../lib/format';

interface DistChartProps {
  values: number[];
  height?: number;
  bins?: number;
  valueMarker?: number | null;
  markerLabel?: string;
  padding?: { t: number; r: number; b: number; l: number };
  color?: string;
}

const DEFAULT_PADDING = { t: 10, r: 10, b: 22, l: 40 };

export function DistChart(props: DistChartProps) {
  return (
    <ParentSize>
      {({ width }) => <DistChartInner {...props} width={Math.max(200, width)} />}
    </ParentSize>
  );
}

function DistChartInner({
  values, height = 160, bins = 40,
  valueMarker = null, markerLabel = 'obs',
  padding = DEFAULT_PADDING, color = 'var(--accent)',
  width,
}: DistChartProps & { width: number }) {
  const w = width;
  const p = padding;
  const plotW = w - p.l - p.r;
  const plotH = height - p.t - p.b;

  if (!values.length) return <svg className="chart" width={w} height={height} />;

  const mn = Math.min(...values);
  const mx = Math.max(...values);
  const span = mx - mn || 1;
  const edges: number[] = [];
  for (let i = 0; i <= bins; i++) edges.push(mn + span * i / bins);
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    let b = Math.floor((v - mn) / span * bins);
    if (b >= bins) b = bins - 1;
    if (b < 0) b = 0;
    counts[b]++;
  }
  const cmax = Math.max(...counts) || 1;

  const x = (v: number) => p.l + (v - mn) / span * plotW;
  const y = (c: number) => p.t + (1 - c / cmax) * plotH;

  const sorted = [...values].sort((a, b) => a - b);
  const quantiles = [0.025, 0.5, 0.975].map((q) => sorted[Math.floor(q * sorted.length)]);

  return (
    <svg className="chart" width={w} height={height}>
      {counts.map((c, i) => {
        const xa = x(edges[i]);
        const xb = x(edges[i + 1]);
        return (
          <rect key={i} x={xa} y={y(c)}
            width={Math.max(1, xb - xa - 1)} height={y(0) - y(c)}
            fill={color} opacity={0.7} />
        );
      })}
      {valueMarker != null && isFinite(valueMarker) && (
        <g>
          <line x1={x(valueMarker)} x2={x(valueMarker)} y1={p.t} y2={height - p.b}
            stroke="var(--warn)" strokeWidth={1.2} />
          <text className="axis" x={x(valueMarker) + 4} y={p.t + 10} fill="var(--warn)">
            {markerLabel}: {valueMarker.toFixed(2)}
          </text>
        </g>
      )}
      {quantiles.map((vv, k) => (
        <line key={k} x1={x(vv)} x2={x(vv)} y1={y(0)} y2={y(0) + 4} stroke="var(--muted)" />
      ))}
      {niceTicks(mn, mx, 5).map((v, k) => (
        <text key={k} className="axis" x={x(v)} y={height - p.b + 12} textAnchor="middle">
          {v.toFixed(2)}
        </text>
      ))}
    </svg>
  );
}
