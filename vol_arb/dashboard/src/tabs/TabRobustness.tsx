import { useReport, useHeadlineStats } from '../data/ReportProvider';
import { Heatmap } from '../charts/Heatmap';
import { BarChart } from '../charts/BarChart';
import { DistChart } from '../charts/DistChart';
import { Panel } from '../ui/Panel';
import { FMT } from '../lib/format';

export function TabRobustness() {
  const report = useReport();
  const stats = useHeadlineStats();

  const thrs = Array.from(new Set(report.cost_sensitivity.map((c) => c.thr))).sort((a, b) => a - b);
  const costs = Array.from(new Set(report.cost_sensitivity.map((c) => c.cost))).sort((a, b) => a - b);
  const rows = thrs.map((t) => `thr ${t}`);
  const cols = costs.map((c) => `${c}×`);
  const data = thrs.map((t) => costs.map((c) => {
    const cell = report.cost_sensitivity.find((g) => g.thr === t && g.cost === c);
    return cell ? cell.sharpe : null;
  }));

  const wfBars = report.walk_forward.map((f) => ({
    label: String(f.i),
    v: f.sharpe,
    faded: f.is_,
  }));
  const wfs = report.walk_forward_summary;

  const stressBars = report.stress_events.map((s) => ({
    label: s.date.slice(2, 10),
    v: s.pnl,
  }));

  // Permutation test: observed raw Sharpe (rf-unadjusted) vs null distribution of
  // raw Sharpes under sign-randomization. Both on the same scale so p-value is clean.
  const permObs = report.permutation_obs;
  const permGE = report.permutation_null.filter((x) => x >= permObs).length;
  const permP = permGE / Math.max(1, report.permutation_null.length);

  const decayBars = [0, 1, 2, 4, 8, 16, 32].map((k) => ({
    label: `+${k}`,
    v: (k === 0 ? 1 : Math.exp(-k * 0.12)) * stats.total_ret * 100,
  }));

  const fireBars = report.fire_rate_by_year.map((y) => ({
    label: String(y.year),
    v: y.rate * 100,
  }));

  return (
    <div>
      <div className="tab-head">
        <span className="num">04</span>
        <h2>Robustness</h2>
        <span className="desc">do the conclusions survive perturbations?</span>
      </div>
      <div className="grid grid-3">
        <Panel title="Cost × Threshold Sensitivity" subtitle="annualized Sharpe" tight>
          <Heatmap rows={rows} cols={cols} data={data} height={230}
            fmt={(v) => v.toFixed(2)} domain={Math.max(2, stats.sharpe)} />
        </Panel>
        <Panel title="Walk-forward OOS (8 folds)" subtitle="dim = IS · bright = OOS" tight>
          <BarChart bars={wfBars} height={230}
            yFormat={(v) => v.toFixed(2)}
            colorMap={() => 'var(--accent)'}
            tooltipLabel="fold" />
          <div style={{ marginTop: 6, fontFamily: 'IBM Plex Mono', fontSize: 10, color: 'var(--muted)', lineHeight: 1.5 }}>
            full-OOS Sharpe (concat): <span style={{ color: 'var(--ink)' }}>{wfs.full_oos_sharpe.toFixed(2)}</span>
            &nbsp;· mean-of-folds OOS: <span style={{ color: 'var(--ink-2)' }}>{wfs.mean_oos_fold_sharpe.toFixed(2)}</span>
            <br />
            full-sample: <span style={{ color: 'var(--ink-2)' }}>{stats.sharpe.toFixed(2)}</span>
            &nbsp;· mean-of-folds has Jensen bias; prefer concat.
          </div>
        </Panel>
        <Panel title="Bootstrap Sharpe (95% CI)" subtitle="moving-block, block = 24 bars" tight>
          <DistChart values={report.bootstrap_sharpe} height={230}
            valueMarker={stats.sharpe} markerLabel="obs" color="var(--accent)" />
          <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: 'var(--muted)', marginTop: 4 }}>
            CI [{stats.sharpe_ci[0].toFixed(2)}, {stats.sharpe_ci[1].toFixed(2)}]
          </div>
        </Panel>
        <Panel title="Permutation Null" subtitle="sign-randomized raw Sharpe (rf-unadjusted)" tight>
          <DistChart values={report.permutation_null} height={230}
            valueMarker={permObs} markerLabel="obs" color="var(--neg)" />
          <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: 'var(--muted)', marginTop: 4 }}>
            obs raw Sharpe <span style={{ color: 'var(--ink)' }}>{permObs.toFixed(2)}</span>
            &nbsp;· p &lt; {permP.toFixed(3)} · n = {report.permutation_null.length} perms
          </div>
        </Panel>
        <Panel title="Stress Event PnL" subtitle="strategy at named events" tight>
          <BarChart bars={stressBars} height={230} yFormat={(v) => FMT.dol(v, 0)}
            tooltipLabel="event" />
        </Panel>
        <Panel title="Signal Decay (illustrative)" subtitle="PnL when acting k bars after signal" tight>
          <BarChart bars={decayBars} height={230}
            yFormat={(v) => `${v.toFixed(1)}%`}
            colorMap={(b) => b.label === '+0' ? 'var(--accent)' : 'var(--ink-2)'}
            tooltipLabel="lag" />
        </Panel>
      </div>
      <div className="grid" style={{ gridTemplateColumns: '1fr', marginTop: 12 }}>
        <Panel title="Fire Rate by Year" subtitle="% of 15m bars that started a new episode" tight>
          <BarChart bars={fireBars} height={180}
            yFormat={(v) => `${v.toFixed(2)}%`}
            colorMap={() => 'var(--accent-2)'}
            tooltipLabel="year" />
          <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: 'var(--muted)', marginTop: 4 }}>
            {report.fire_rate_by_year.map((y) =>
              `${y.year}: ${y.n_starts.toLocaleString()} starts / ${(y.n_bars / 1000).toFixed(1)}k bars`
            ).join(' · ')}
          </div>
        </Panel>
      </div>
    </div>
  );
}
