import { useReport } from '../data/ReportProvider';
import { CalendarHeat } from '../charts/CalendarHeat';
import { BarChart } from '../charts/BarChart';
import { DistChart } from '../charts/DistChart';
import { Panel } from '../ui/Panel';
import { FMT } from '../lib/format';

export function TabRegime() {
  const report = useReport();

  const byReg: Record<string, { n: number; sumEdge: number; pnl: number }> = {};
  for (const e of report.episodes) {
    const reg = e.regime;
    if (!byReg[reg]) byReg[reg] = { n: 0, sumEdge: 0, pnl: 0 };
    byReg[reg].n += 1;
    byReg[reg].sumEdge += e.edge_entry;
    byReg[reg].pnl += e.net_pnl;
  }
  const regBars = Object.entries(byReg).map(([k, v]) => ({ label: k, v: v.pnl, n: v.n }));
  const regEdgeBars = Object.entries(byReg).map(([k, v]) => ({
    label: k, v: v.sumEdge / v.n, n: v.n,
  }));

  const dur = report.episodes.map((e) => e.dur);

  return (
    <div>
      <div className="tab-head">
        <span className="num">05</span>
        <h2>Regime &amp; Timing</h2>
        <span className="desc">when does the edge pay, and when does it disappear?</span>
      </div>
      <div className="grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <Panel title="Monthly Returns — Strategy" subtitle="green = positive · red = negative" tight>
          <CalendarHeat monthData={report.monthly_strat} height={170}
            fmt={(v) => `${(v * 100).toFixed(1)}%`} />
        </Panel>
        <Panel title="Monthly Returns — Benchmark (60/40)" subtitle="same scale" tight>
          <CalendarHeat monthData={report.monthly_bench} height={170}
            fmt={(v) => `${(v * 100).toFixed(1)}%`} />
        </Panel>
      </div>
      <div className="grid grid-3">
        <Panel title="PnL by Regime" subtitle="rich · cheap · neutral · stress" tight>
          <BarChart bars={regBars} height={200} yFormat={(v) => FMT.dol(v, 0)}
            tooltipLabel="regime" />
        </Panel>
        <Panel title="Avg Entry Edge by Regime" subtitle="vol pts" tight>
          <BarChart bars={regEdgeBars} height={200} yFormat={(v) => v.toFixed(1)}
            tooltipLabel="regime" />
        </Panel>
        <Panel title="Hold-duration Distribution" subtitle="bars · 1h each" tight>
          <DistChart values={dur} height={200} bins={30} />
        </Panel>
      </div>
    </div>
  );
}
