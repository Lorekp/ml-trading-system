import { ParentSize } from '@visx/responsive';
import { useState } from 'react';
import { niceTicks, fmtDate } from '../lib/format';
import { showTip, hideTip, tooltipHeader, tooltipRow } from '../ui/Tooltip';
import { useCrosshair } from './CrosshairContext';

export interface LineSeries {
  label: string;
  y: (number | null)[];
  color: string;
  width?: number;
  fillArea?: boolean;
  fmt?: (v: number) => string;
}

interface BenchSeries {
  label: string;
  y: (number | null)[];
  color: string;
  fmt?: (v: number) => string;
}

interface Annotation { i: number; label: string }

interface LineChartProps {
  series: LineSeries[];
  benchmarkSeries?: BenchSeries | null;
  height?: number;
  yFormat?: (v: number) => string;
  padding?: { t: number; r: number; b: number; l: number };
  annotations?: Annotation[];
  showGrid?: boolean;
  crosshair?: boolean;
  /** Index → label for x-axis. If absent, integer indices are used. */
  xLabel?: (i: number) => string;
  /** 'log' uses log10 scale; non-positive values fall back to linear. */
  yScale?: 'linear' | 'log';
}

const DEFAULT_PADDING = { t: 10, r: 12, b: 22, l: 52 };

function clip01(x: number) {
  return Math.max(0, Math.min(1, x));
}

function logTicks(lo: number, hi: number): number[] {
  if (lo <= 0 || hi <= 0) return [];
  const a = Math.floor(Math.log10(lo));
  const b = Math.ceil(Math.log10(hi));
  const out: number[] = [];
  for (let d = a; d <= b; d++) {
    const v = Math.pow(10, d);
    if (v >= lo && v <= hi) out.push(v);
  }
  // If only 1-2 decades visible, add 2x and 5x subticks.
  if (out.length <= 2) {
    const extra: number[] = [];
    for (let d = a; d <= b; d++) {
      for (const m of [2, 5]) {
        const v = m * Math.pow(10, d);
        if (v >= lo && v <= hi) extra.push(v);
      }
    }
    return [...out, ...extra].sort((x, y) => x - y);
  }
  return out;
}

export function LineChart(props: LineChartProps) {
  return (
    <ParentSize>
      {({ width }) => <LineChartInner {...props} width={Math.max(200, width)} />}
    </ParentSize>
  );
}

function LineChartInner({
  series, benchmarkSeries, height = 220, yFormat = (v) => v.toFixed(2),
  padding = DEFAULT_PADDING, annotations = [], showGrid = true, crosshair = true,
  xLabel, width, yScale = 'linear',
}: LineChartProps & { width: number }) {
  const w = width;
  const h = height;
  const p = padding;
  const plotW = w - p.l - p.r;
  const plotH = h - p.t - p.b;

  const cross = useCrosshair();
  const [localHover, setLocalHover] = useState<number | null>(null);
  const hover = cross.cursor ?? localHover;

  const all: number[] = [];
  for (const s of series) for (const v of s.y) if (v != null && isFinite(v)) all.push(v);
  if (benchmarkSeries) for (const v of benchmarkSeries.y) if (v != null && isFinite(v)) all.push(v);
  if (!all.length) return <svg className="chart" width={w} height={h} />;

  // Log scale falls back to linear if any series value is ≤ 0.
  const isLog = yScale === 'log' && all.every((v) => v > 0);
  const mapY = (v: number) => (isLog ? Math.log10(v) : v);

  let yMin = Math.min(...all);
  let yMax = Math.max(...all);
  if (isLog) {
    // In log space, pad by 5% of the log-range.
    const lo = Math.log10(yMin);
    const hi = Math.log10(yMax);
    const padL = (hi - lo) * 0.05 || 0.05;
    yMin = Math.pow(10, lo - padL);
    yMax = Math.pow(10, hi + padL);
  } else {
    const pad = (yMax - yMin) * 0.08 || 1;
    yMin -= pad;
    yMax += pad;
  }

  const xLen = series[0].y.length;
  const x = (i: number) => p.l + (xLen <= 1 ? plotW / 2 : (i / (xLen - 1)) * plotW);
  const y = (v: number) => p.t + (1 - (mapY(v) - mapY(yMin)) / (mapY(yMax) - mapY(yMin))) * plotH;

  const yt = isLog
    ? logTicks(yMin, yMax)
    : niceTicks(yMin, yMax, 5);
  const xTicks: { i: number; label: string }[] = [];
  const nT = 6;
  for (let k = 0; k <= nT; k++) {
    const i = Math.round((k * (xLen - 1)) / nT);
    xTicks.push({ i, label: xLabel ? xLabel(i) : String(i) });
  }

  const path = (yy: (number | null)[]) => {
    const cmds: string[] = [];
    let started = false;
    for (let i = 0; i < yy.length; i++) {
      const v = yy[i];
      if (v == null || !isFinite(v)) {
        started = false;
        continue;
      }
      cmds.push(`${started ? 'L' : 'M'} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`);
      started = true;
    }
    return cmds.join(' ');
  };

  const areaPath = (yy: (number | null)[]) => {
    const top = path(yy);
    if (!top) return '';
    return `${top} L ${x(yy.length - 1).toFixed(1)} ${y(yMin).toFixed(1)} L ${x(0).toFixed(1)} ${y(yMin).toFixed(1)} Z`;
  };

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    if (!crosshair) return;
    const r = e.currentTarget.getBoundingClientRect();
    const xp = e.clientX - r.left;
    const ratio = (xp - p.l) / plotW;
    const i = Math.round(clip01(ratio) * (xLen - 1));
    setLocalHover(i);
    cross.setCursor(i);
    let html = tooltipHeader(xLabel ? xLabel(i) : String(i));
    for (const s of series) {
      const v = s.y[i];
      if (v == null) continue;
      html += tooltipRow(s.color, s.label, s.fmt ? s.fmt(v) : yFormat(v));
    }
    if (benchmarkSeries) {
      const v = benchmarkSeries.y[i];
      if (v != null) html += tooltipRow(benchmarkSeries.color, benchmarkSeries.label, benchmarkSeries.fmt ? benchmarkSeries.fmt(v) : yFormat(v));
    }
    showTip(html, e);
  }

  function onLeave() {
    setLocalHover(null);
    cross.setCursor(null);
    hideTip();
  }

  return (
    <svg className="chart" width={w} height={h} onMouseMove={onMove} onMouseLeave={onLeave}>
      {showGrid && yt.map((v, k) => (
        <line key={k} x1={p.l} x2={w - p.r} y1={y(v)} y2={y(v)}
          stroke="var(--rule)" strokeWidth={1}
          strokeDasharray={v === 0 ? '' : '2 3'} />
      ))}
      {yt.map((v, k) => (
        <text key={k} className="axis" x={p.l - 6} y={y(v) + 3} textAnchor="end">
          {yFormat(v)}
        </text>
      ))}
      {xTicks.map((t, k) => (
        <g key={k}>
          <line x1={x(t.i)} x2={x(t.i)} y1={p.t} y2={h - p.b}
            stroke="var(--rule)" strokeWidth={1} strokeDasharray="2 3"
            opacity={showGrid ? 0.5 : 0} />
          <text className="axis" x={x(t.i)} y={h - p.b + 12} textAnchor="middle">
            {t.label}
          </text>
        </g>
      ))}
      {benchmarkSeries && (
        <path d={path(benchmarkSeries.y)} stroke={benchmarkSeries.color} strokeWidth={1}
          fill="none" strokeDasharray="3 3" opacity={0.85} />
      )}
      {series.map((s, k) => (
        <g key={k}>
          {s.fillArea && <path d={areaPath(s.y)} fill={s.color} opacity={0.14} />}
          <path d={path(s.y)} stroke={s.color} strokeWidth={s.width ?? 1.4} fill="none" />
        </g>
      ))}
      {annotations.map((a, k) => (
        <g key={k}>
          <line x1={x(a.i)} x2={x(a.i)} y1={p.t} y2={h - p.b}
            stroke="var(--warn)" strokeWidth={0.8} strokeDasharray="2 2" opacity={0.7} />
          <text x={x(a.i) + 4} y={p.t + 10} fill="var(--warn)"
            style={{ fontSize: '9px', fontFamily: 'IBM Plex Mono' }}>
            {a.label}
          </text>
        </g>
      ))}
      {hover != null && hover >= 0 && hover < xLen && crosshair && (
        <g>
          <line x1={x(hover)} x2={x(hover)} y1={p.t} y2={h - p.b} className="crosshair-v" />
          {series.map((s, k) => {
            const v = s.y[hover];
            return v == null ? null : (
              <circle key={k} cx={x(hover)} cy={y(v)} r={3}
                fill={s.color} stroke="var(--bg)" strokeWidth={1} />
            );
          })}
        </g>
      )}
    </svg>
  );
}
