import { ParentSize } from '@visx/responsive';
import { useState } from 'react';
import { niceTicks } from '../lib/format';
import { showTip, hideTip, tooltipHeader, tooltipRow } from '../ui/Tooltip';
import { useCrosshair } from './CrosshairContext';

export interface StackLayer {
  label: string;
  color: string;
  y: number[];
}

interface StackedAreaProps {
  xLen: number;
  layers: StackLayer[];
  height?: number;
  yFormat?: (v: number) => string;
  padding?: { t: number; r: number; b: number; l: number };
  xLabel?: (i: number) => string;
}

const DEFAULT_PADDING = { t: 10, r: 12, b: 22, l: 62 };

function clip01(x: number) {
  return Math.max(0, Math.min(1, x));
}

export function StackedArea(props: StackedAreaProps) {
  return (
    <ParentSize>
      {({ width }) => <StackedAreaInner {...props} width={Math.max(200, width)} />}
    </ParentSize>
  );
}

function StackedAreaInner({
  xLen, layers, height = 260,
  yFormat = (v) => v.toFixed(0),
  padding = DEFAULT_PADDING, xLabel,
  width,
}: StackedAreaProps & { width: number }) {
  const w = width;
  const p = padding;
  const plotW = w - p.l - p.r;
  const plotH = height - p.t - p.b;

  const cross = useCrosshair();
  const [localHover, setLocalHover] = useState<number | null>(null);
  const hover = cross.cursor ?? localHover;

  if (!xLen || !layers.length) return <svg className="chart" width={w} height={height} />;

  const layerBands = layers.map((L) => ({
    label: L.label,
    color: L.color,
    y0: new Array(xLen),
    y1: new Array(xLen),
  }));
  const pos = new Array(xLen).fill(0);
  const neg = new Array(xLen).fill(0);
  for (let i = 0; i < xLen; i++) {
    let pa = 0, na = 0;
    layers.forEach((L, k) => {
      const v = L.y[i] ?? 0;
      if (v >= 0) {
        layerBands[k].y0[i] = pa;
        layerBands[k].y1[i] = pa + v;
        pa += v;
      } else {
        layerBands[k].y0[i] = na + v;
        layerBands[k].y1[i] = na;
        na += v;
      }
    });
    pos[i] = pa;
    neg[i] = na;
  }

  const yMin = Math.min(0, ...neg) * 1.08;
  const yMax = Math.max(0, ...pos) * 1.08 || 1;
  const xP = (i: number) => p.l + (xLen <= 1 ? plotW / 2 : (i / (xLen - 1)) * plotW);
  const yP = (v: number) => p.t + (1 - (v - yMin) / (yMax - yMin || 1)) * plotH;
  const yt = niceTicks(yMin, yMax, 5);
  const stepX = Math.max(1, Math.floor(xLen / 800));

  function path(band: typeof layerBands[number]): string {
    const pts: string[] = [];
    for (let i = 0; i < xLen; i += stepX) {
      pts.push(`${i === 0 ? 'M' : 'L'} ${xP(i).toFixed(1)} ${yP(band.y1[i]).toFixed(1)}`);
    }
    for (let i = xLen - 1; i >= 0; i -= stepX) {
      pts.push(`L ${xP(i).toFixed(1)} ${yP(band.y0[i]).toFixed(1)}`);
    }
    pts.push('Z');
    return pts.join(' ');
  }

  const xTicks: { i: number; label: string }[] = [];
  for (let k = 0; k <= 6; k++) {
    const i = Math.round((k * (xLen - 1)) / 6);
    xTicks.push({ i, label: xLabel ? xLabel(i) : String(i) });
  }

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    const xpos = e.clientX - r.left;
    const ratio = (xpos - p.l) / plotW;
    const i = Math.round(clip01(ratio) * (xLen - 1));
    setLocalHover(i);
    cross.setCursor(i);
    let html = tooltipHeader(xLabel ? xLabel(i) : String(i));
    layers.forEach((L) => {
      html += tooltipRow(L.color, L.label, yFormat(L.y[i] ?? 0));
    });
    showTip(html, e);
  }

  function onLeave() {
    setLocalHover(null);
    cross.setCursor(null);
    hideTip();
  }

  return (
    <svg className="chart" width={w} height={height} onMouseMove={onMove} onMouseLeave={onLeave}>
      {yt.map((v, k) => (
        <g key={k}>
          <line x1={p.l} x2={w - p.r} y1={yP(v)} y2={yP(v)}
            stroke="var(--rule)" strokeDasharray={v === 0 ? '' : '2 3'} />
          <text className="axis" x={p.l - 6} y={yP(v) + 3} textAnchor="end">
            {yFormat(v)}
          </text>
        </g>
      ))}
      {xTicks.map((t, k) => (
        <text key={k} className="axis" x={xP(t.i)} y={height - p.b + 12} textAnchor="middle">
          {t.label}
        </text>
      ))}
      {layerBands.map((band, k) => (
        <path key={k} d={path(band)} fill={band.color} opacity={0.75} />
      ))}
      {hover != null && hover >= 0 && hover < xLen && (
        <line x1={xP(hover)} x2={xP(hover)} y1={p.t} y2={height - p.b} className="crosshair-v" />
      )}
    </svg>
  );
}
