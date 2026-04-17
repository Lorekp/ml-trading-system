import { useReport, useControls, useBars, useDates } from '../data/ReportProvider';
import { StackedArea } from '../charts/StackedArea';
import { BarChart } from '../charts/BarChart';
import { Panel, Chip } from '../ui/Panel';
import { FMT, fmtDate } from '../lib/format';

const COLORS = {
  vega: '#14b8a6',
  theta: '#8b5cf6',
  hedge: '#60a5fa',
  costs: '#ef4444',
};

function cumsum(arr: number[]): number[] {
  const out = new Array(arr.length);
  let s = 0;
  for (let i = 0; i < arr.length; i++) {
    s += arr[i];
    out[i] = s;
  }
  return out;
}

export function TabAttribution() {
  const report = useReport();
  const controls = useControls();
  const bars = useBars();
  const dates = useDates();
  const [a, b] = controls.range;

  const sliceCum = (arr: number[]) => cumsum(arr.slice(a, b + 1));
  const layers = [
    { label: 'Vega (IV)', color: COLORS.vega, y: sliceCum(bars.vega_pnl_bar) },
    { label: 'Theta', color: COLORS.theta, y: sliceCum(bars.theta_pnl_bar) },
    { label: 'Delta hedge', color: COLORS.hedge, y: sliceCum(bars.delta_hedge_pnl_bar) },
    { label: 'Costs', color: COLORS.costs, y: sliceCum(bars.costs_bar) },
  ];

  const xLabel = (i: number) => fmtDate(dates[a + i]);

  // Episode entry/exit are indices into the 15m source parquet; a/b are indices into
  // the 1h-downsampled display. Filter by timestamp against dates[a..b].
  const tStart = dates[a] ? dates[a].getTime() : 0;
  const tEnd = dates[b] ? dates[b].getTime() : tStart + 1;
  const epR = report.episodes.filter((e) => {
    const t = new Date(e.ts_entry).getTime();
    return t >= tStart && t <= tEnd + 3600_000;
  });
  // Dynamic time-bucketing: sum net_pnl into ~100 equal-time buckets across the visible range.
  const nBuckets = Math.min(100, Math.max(20, epR.length));
  const bucketW = (tEnd - tStart) / nBuckets;
  const buckets = new Array(nBuckets).fill(0).map((_, i) => ({
    t: tStart + i * bucketW,
    pnl: 0,
  }));
  for (const e of epR) {
    const t = new Date(e.ts_entry).getTime();
    const idx = Math.min(nBuckets - 1, Math.max(0, Math.floor((t - tStart) / bucketW)));
    buckets[idx].pnl += e.net_pnl;
  }
  const epBars = buckets.map((b_) => ({
    label: fmtDate(new Date(b_.t)),
    v: b_.pnl,
  }));

  const totalHedge = epR.reduce((s, e) => s + e.hedge_costs, 0);
  const totalFees = epR.reduce((s, e) => s + e.option_fees, 0);
  const totalSlip = epR.reduce((s, e) => s + e.slippage, 0);
  const costBars = [
    { label: 'hedge', v: totalHedge },
    { label: 'fees', v: totalFees },
    { label: 'slip', v: totalSlip },
  ];

  const longPnl = epR.filter((e) => e.side > 0).reduce((s, e) => s + e.net_pnl, 0);
  const shortPnl = epR.filter((e) => e.side < 0).reduce((s, e) => s + e.net_pnl, 0);

  return (
    <div>
      <div className="tab-head">
        <span className="num">02</span>
        <h2>PnL Attribution</h2>
        <span className="desc">vega · theta · delta hedge · costs — mode = {controls.mode}</span>
      </div>
      <Panel
        title="Cumulative PnL by Leg"
        subtitle={
          <>
            {layers.map((L) => <Chip key={L.label} color={L.color} label={L.label} />)}
          </>
        }
        tight>
        <StackedArea xLen={layers[0].y.length} layers={layers} height={280}
          yFormat={(v) => FMT.dol(v, 0)}
          xLabel={xLabel} />
      </Panel>
      <div className="grid grid-3">
        <Panel title="Episode PnL (chron)" subtitle={`${epR.length.toLocaleString()} episodes → ${nBuckets} time-buckets · sum $`} tight>
          <BarChart bars={epBars} height={220} yFormat={(v) => FMT.dol(v, 0)}
            labels={false} tooltipLabel="bucket"
            colorMap={(b_) => b_.v >= 0 ? 'var(--pos)' : 'var(--neg)'} />
        </Panel>
        <Panel title="Cost Breakdown" subtitle="range-summed, negative = paid" tight>
          <BarChart bars={costBars} height={220} yFormat={(v) => FMT.dol(v, 0)} />
        </Panel>
        <Panel title="Long vs Short Vol" subtitle="net $ contribution" tight>
          <BarChart
            bars={[
              { label: 'long vol', v: longPnl },
              { label: 'short vol', v: shortPnl },
            ]}
            height={220}
            yFormat={(v) => FMT.dol(v, 0)}
            colorMap={(b) => b.label.includes('long') ? 'var(--accent)' : 'var(--accent-2)'} />
        </Panel>
      </div>
    </div>
  );
}
