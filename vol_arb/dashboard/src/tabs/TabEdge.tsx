import { useReport, useControls, useDates } from '../data/ReportProvider';
import { BarChart } from '../charts/BarChart';
import { Scatter } from '../charts/Scatter';
import { LineChart } from '../charts/LineChart';
import { DistChart } from '../charts/DistChart';
import { Panel } from '../ui/Panel';
import { FMT } from '../lib/format';

export function TabEdge() {
  const report = useReport();
  const controls = useControls();
  const [a, b] = controls.range;
  const dates = useDates();

  const decBars = report.decile_table.map((d) => ({
    label: String(d.d),
    v: d.mean,
    se: d.se,
    n: d.n,
  }));

  const tStart = dates[a] ? dates[a].getTime() : -Infinity;
  const tEnd = dates[b] ? dates[b].getTime() + 3600_000 : Infinity;
  const epR = report.episodes.filter((e) => {
    const t = new Date(e.ts_entry).getTime();
    return t >= tStart && t <= tEnd;
  });
  const stride = Math.max(1, Math.floor(epR.length / 1000));
  const scatterPts = epR
    .filter((_, i) => i % stride === 0)
    .map((e) => ({
      x: e.edge_entry,
      y: e.N_contracts > 0 ? e.net_pnl / e.N_contracts : 0,
      r: 2,
      label: `#${e.id} · ${e.side > 0 ? 'long' : 'short'}`,
    }));

  const icYs = report.rolling_ic.map((d) => d.ic);
  const edgeArr = report.bars_common.edge.slice(a, b + 1);
  const edgeStride = Math.max(1, Math.floor(edgeArr.length / 5000));
  const edgeDist = edgeArr.filter((_, i) => i % edgeStride === 0);

  const triggered = report.episodes.length;
  const total = report.meta.n_bars;

  return (
    <div>
      <div className="tab-head">
        <span className="num">03</span>
        <h2>Edge &amp; Signal</h2>
        <span className="desc">is the signal monotonic? is it stable?</span>
      </div>
      <Panel
        title="PnL per Contract by Edge Decile"
        subtitle="D1 = most negative edge (long vol) · D10 = most positive (short vol) · ±1 SE"
        tight>
        <BarChart bars={decBars} height={280}
          yFormat={(v) => FMT.dol(v, 0)} errorBars
          tooltipLabel="decile"
          colorMap={(b) => b.v >= 0 ? 'var(--accent)' : 'var(--accent-2)'} />
      </Panel>
      <div className="grid grid-3">
        <Panel title="Edge → PnL/contract (scatter)" subtitle={`n = ${scatterPts.length}`} tight>
          <Scatter points={scatterPts} height={200}
            xLabel="edge (iv − rv̂, vol pts)" yLabel="pnl/contract ($)"
            yFormat={(v) => FMT.dol(v, 0)}
            regression
            colorBy={(d) => d.label?.includes('long') ? 'var(--accent)' : 'var(--accent-2)'} />
        </Panel>
        <Panel title="Rolling 180d IC" subtitle="|edge| vs forward 5-bar return" tight>
          <LineChart height={200}
            series={[{ label: 'IC', y: icYs, color: 'var(--accent)' }]}
            yFormat={(v) => v.toFixed(2)}
            xLabel={(i) => String(i)} />
        </Panel>
        <Panel title="Edge Distribution" subtitle="threshold band ±7 vol pts" tight>
          <DistChart values={edgeDist} height={200} bins={40}
            valueMarker={7} markerLabel="thr" />
          <div style={{
            display: 'flex', justifyContent: 'space-between',
            fontFamily: 'IBM Plex Mono', fontSize: 10, color: 'var(--muted)',
            marginTop: 8, padding: '0 4px',
          }}>
            <span>triggered: {triggered.toLocaleString()}</span>
            <span>total bars: {total.toLocaleString()}</span>
            <span>fire rate: {((triggered / total) * 100).toFixed(2)}%</span>
          </div>
        </Panel>
      </div>
    </div>
  );
}
