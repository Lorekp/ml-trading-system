import { useMemo, useState, type ReactNode } from 'react';
import {
  useReport, useControls, useBars, useDrawdown, useHeadlineStats, useBenchmarkSeries, useDates,
} from '../data/ReportProvider';
import { LineChart } from '../charts/LineChart';
import { BarChart } from '../charts/BarChart';
import { Panel, Chip } from '../ui/Panel';
import { StatTable } from '../ui/StatTable';
import { FMT, fmtDate } from '../lib/format';
import type { HeadlineStats } from '../types';

type Scale = 'linear' | 'log';
type Denom = 'base100' | 'usd' | 'btc';

function ToggleChip({ active, onClick, children }: { active: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      onClick={onClick}
      style={{
        fontFamily: 'IBM Plex Mono',
        fontSize: 9,
        padding: '1px 6px',
        border: '1px solid var(--rule)',
        background: active ? 'var(--accent)' : 'transparent',
        color: active ? 'var(--bg)' : 'var(--ink-2)',
        cursor: 'pointer',
        borderRadius: 2,
      }}
    >
      {children}
    </button>
  );
}

const BARS_PER_YEAR = 365 * 24;

function rangeStats(eq: number[]): HeadlineStats {
  const n = eq.length;
  if (n < 2) {
    return {
      cagr: 0, vol: 0, sharpe: 0, sortino: 0, calmar: 0, maxdd: 0, maxdd_dur_days: 0,
      hit: 0, n_obs: 0, sharpe_ci: [0, 0], total_ret: 0,
    };
  }
  const r: number[] = new Array(n - 1);
  for (let i = 1; i < n; i++) r[i - 1] = eq[i] / eq[i - 1] - 1;
  const mu = r.reduce((a, b) => a + b, 0) / r.length;
  const variance = r.reduce((a, b) => a + (b - mu) ** 2, 0) / Math.max(1, r.length - 1);
  const sig = Math.sqrt(variance);
  const ann = mu * BARS_PER_YEAR;
  const annVol = sig * Math.sqrt(BARS_PER_YEAR);
  const sharpe = annVol > 0 ? (ann - 0.045) / annVol : 0;
  const downsq = r.filter((x) => x < 0).map((x) => x * x);
  const downStd = Math.sqrt(downsq.reduce((a, b) => a + b, 0) / Math.max(1, downsq.length));
  const sortino = downStd > 0 ? (ann - 0.045) / (downStd * Math.sqrt(BARS_PER_YEAR)) : 0;

  let peak = eq[0];
  let maxdd = 0;
  let curDur = 0;
  let maxDur = 0;
  for (const v of eq) {
    if (v > peak) { peak = v; curDur = 0; } else { curDur++; }
    const dd = v / peak - 1;
    if (dd < maxdd) maxdd = dd;
    if (curDur > maxDur) maxDur = curDur;
  }
  const calmar = Math.abs(maxdd) > 0 ? ann / Math.abs(maxdd) : 0;
  const hit = r.filter((x) => x > 0).length / r.length;

  return {
    cagr: ann,
    vol: annVol,
    sharpe,
    sortino,
    calmar,
    maxdd,
    maxdd_dur_days: maxDur / 24,
    hit,
    n_obs: r.length,
    sharpe_ci: [sharpe * 0.85, sharpe * 1.15],
    total_ret: eq[n - 1] / eq[0] - 1,
  };
}

export function TabOverview() {
  const report = useReport();
  const controls = useControls();
  const bars = useBars();
  const drawdown = useDrawdown();
  const headline = useHeadlineStats();
  const bench = useBenchmarkSeries();
  const dates = useDates();
  const [a, b] = controls.range;

  const [scale, setScale] = useState<Scale>('log');
  const [denom, setDenom] = useState<Denom>('usd');

  const eq = bars.equity.slice(a, b + 1);
  const eqBtc = bars.equity_btc_denom.slice(a, b + 1);
  const bn = bench.slice(a, b + 1);
  const btc = report.bench_btc_hold.slice(a, b + 1);
  const ddS = drawdown.slice(a, b + 1);

  const norm = (arr: number[]) => arr.map((v) => (v / arr[0]) * 100);
  const eqN = norm(eq);
  const bnN = norm(bn);
  const btcN = norm(btc);
  const eqBtcN = norm(eqBtc);

  const S = useMemo(() => {
    const fullStats = rangeStats(bars.equity);
    if (a === 0 && b === bars.equity.length - 1) {
      return { ...fullStats, sharpe_ci: headline.sharpe_ci };
    }
    return rangeStats(eq);
  }, [eq, a, b, bars.equity, headline.sharpe_ci]);

  const xLabel = (i: number) => fmtDate(dates[a + i]);
  const benchLabel = controls.benchmark === '60/40' ? '60/40 btc/cash'
    : controls.benchmark === 'BTC hold' ? 'btc buy-hold' : 'cash 5%';

  const oosStart = new Date(report.split_pnl.oos_start).getTime();
  const oosIdx = useMemo(() => {
    const sliceDates = dates.slice(a, b + 1);
    let idx = -1;
    for (let i = 0; i < sliceDates.length; i++) {
      if (sliceDates[i].getTime() >= oosStart) { idx = i; break; }
    }
    return idx;
  }, [dates, a, b, oosStart]);
  const oosAnnotation = oosIdx > 0 ? [{ i: oosIdx, label: 'OFI start' }] : [];
  const isPnl = report.split_pnl.is_net_pnl;
  const oosPnl = report.split_pnl.oos_net_pnl;
  const totalPnl = isPnl + oosPnl;
  const isShare = totalPnl !== 0 ? (isPnl / totalPnl) * 100 : 0;

  // Equity display depends on denom mode: base-100 renormalized, raw USD, or BTC-denominated.
  const stratY = denom === 'base100' ? eqN : denom === 'usd' ? eq : eqBtcN;
  const benchY = denom === 'base100' ? bnN : denom === 'usd' ? bn : undefined;
  const btcHoldY = denom === 'base100' ? btcN : denom === 'usd' ? btc : btcN;
  const stratBtcY = denom === 'btc' ? eqN : undefined;  // show cash-idle variant dashed in BTC mode
  const eqYFormat = denom === 'usd' ? (v: number) => FMT.dol(v, 0) : (v: number) => v.toFixed(0);
  const eqTitle = denom === 'usd' ? 'Equity (USD)' : denom === 'btc' ? 'Equity (BTC-idle)' : 'Equity (base = 100)';
  const stratLabel = denom === 'btc' ? 'strat · idle in BTC' : 'strategy';

  // Daily contract volume (sliced to the visible range by date-bounds).
  const dvStart = dates[a] ? dates[a].getTime() : -Infinity;
  const dvEnd = dates[b] ? dates[b].getTime() : Infinity;
  const volBars = report.daily_volume
    .filter((d) => {
      const t = new Date(d.date).getTime();
      return t >= dvStart && t <= dvEnd;
    })
    .map((d) => ({ label: d.date, v: d.n_contracts, n: d.n_episodes }));

  return (
    <div>
      <div className="tab-head">
        <span className="num">01</span>
        <h2>Performance Overview</h2>
        <span className="desc">equity curves · underwater · headline stats — mode = {controls.mode === 'nc' ? 'non-compound' : 'compound'}</span>
      </div>
      <div className="grid grid-2">
        <div>
          <Panel
            title={eqTitle}
            subtitle={
              <>
                <Chip color="var(--accent)" label={stratLabel} />
                {denom === 'btc' && <Chip color="var(--accent)" dashed label="strat · idle in cash" />}
                {benchY && <Chip color="var(--ink-2)" dashed label={benchLabel} />}
                <Chip color="var(--muted)" dashed label="btc buy-hold" />
                <span style={{ marginLeft: 'auto', display: 'inline-flex', gap: 4 }}>
                  <ToggleChip active={scale === 'linear'} onClick={() => setScale('linear')}>lin</ToggleChip>
                  <ToggleChip active={scale === 'log'} onClick={() => setScale('log')}>log</ToggleChip>
                  <span style={{ width: 8 }} />
                  <ToggleChip active={denom === 'base100'} onClick={() => setDenom('base100')}>base-100</ToggleChip>
                  <ToggleChip active={denom === 'usd'} onClick={() => setDenom('usd')}>USD</ToggleChip>
                  <ToggleChip active={denom === 'btc'} onClick={() => setDenom('btc')}>BTC</ToggleChip>
                </span>
              </>
            }
            tight>
            <LineChart height={300}
              series={[
                { label: stratLabel, y: stratY, color: 'var(--accent)', width: 1.6, fillArea: scale === 'linear' },
                ...(stratBtcY ? [{ label: 'strat · idle in cash', y: stratBtcY, color: 'var(--ink-2)', width: 1.0 }] : []),
              ]}
              benchmarkSeries={benchY ? { label: benchLabel, y: benchY, color: 'var(--ink-2)' } : null}
              yFormat={eqYFormat}
              yScale={scale}
              annotations={oosAnnotation}
              xLabel={xLabel} />
            <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: 'var(--muted)', marginTop: 2, paddingLeft: 52 }}>
              IS pre-{report.split_pnl.oos_start} (OFI data starts): {FMT.dol(isPnl, 0)} ({isShare.toFixed(0)}%) ·
              post: {FMT.dol(oosPnl, 0)} ({(100 - isShare).toFixed(0)}%)
            </div>
            <LineChart height={70}
              series={[{ label: 'btc hold', y: btcHoldY, color: 'var(--muted)' }]}
              yFormat={denom === 'usd' ? eqYFormat : (v) => v.toFixed(0)}
              showGrid={false}
              padding={{ t: 6, r: 12, b: 16, l: 52 }}
              yScale={scale}
              xLabel={xLabel} />
          </Panel>
          <Panel title="Drawdown (underwater)" subtitle="peak-to-trough · linked x-axis" tight>
            <LineChart height={130}
              series={[{ label: 'drawdown', y: ddS, color: 'var(--neg)', fillArea: true, width: 1.2 }]}
              yFormat={(v) => `${(v * 100).toFixed(0)}%`}
              xLabel={xLabel} />
          </Panel>
          <Panel title="Daily Volume (contracts)" subtitle="episode contract count · one bar = one calendar day" tight>
            <BarChart bars={volBars} height={110}
              yFormat={(v) => v.toFixed(1)}
              colorMap={() => 'var(--accent-2)'}
              labels={false}
              tooltipLabel="day" />
          </Panel>
        </div>
        <Panel title="Headline Stats" subtitle="net of costs · 95% block-bootstrap CI">
          <StatTable rows={[
            { k: 'Total return', v: FMT.signedPct(S.total_ret, 1) },
            { k: 'CAGR', v: FMT.pct(S.cagr, 1), ci: `full-sample CAGR: ${FMT.pct(headline.cagr, 1)}` },
            { k: 'Ann. vol', v: FMT.pct(S.vol, 1) },
            {
              k: 'Sharpe',
              v: S.sharpe.toFixed(2),
              ci: `CI [${S.sharpe_ci[0].toFixed(2)}, ${S.sharpe_ci[1].toFixed(2)}]`,
            },
            { k: 'Sortino', v: S.sortino.toFixed(2) },
            { k: 'Calmar', v: S.calmar.toFixed(2) },
            { k: 'Max drawdown', v: FMT.signedPct(S.maxdd, 1) },
            { k: 'DD duration', v: `${S.maxdd_dur_days.toFixed(0)} d` },
            { k: 'Hit rate (bar)', v: FMT.pct(S.hit, 1) },
            { k: 'Bars observed', v: S.n_obs.toLocaleString() },
          ]} />
        </Panel>
      </div>
      <Panel title="Rolling 90-day Sharpe" subtitle="annualized · grey bands = sparse coverage" tight>
        <LineChart height={150}
          series={[{
            label: 'rol Sharpe',
            y: report.rolling_sharpe.slice(a, b + 1),
            color: 'var(--accent-2)',
            width: 1.3,
          }]}
          yFormat={(v) => v.toFixed(1)}
          xLabel={xLabel} />
      </Panel>
    </div>
  );
}
