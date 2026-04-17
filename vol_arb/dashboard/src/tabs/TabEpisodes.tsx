import { useEffect, useMemo, useState } from 'react';
import { useReport, useControls, useDates } from '../data/ReportProvider';
import { LineChart } from '../charts/LineChart';
import { BarChart } from '../charts/BarChart';
import { StackedArea } from '../charts/StackedArea';
import { Panel } from '../ui/Panel';
import { FMT } from '../lib/format';
import type { Episode } from '../types';

type SortKey = 'id' | 'entry' | 'dur' | 'edge_entry' | 'iv_entry' | 'rv_realized' | 'vega_pnl' | 'theta_pnl' | 'net_pnl';
type FilterMode = 'all' | 'winners' | 'losers' | 'long' | 'short';

const FILTERS: FilterMode[] = ['all', 'winners', 'losers', 'long', 'short'];

export function TabEpisodes() {
  const report = useReport();
  const controls = useControls();
  const [a, b] = controls.range;
  const dates = useDates();
  const ep = useMemo(() => {
    const tStart = dates[a] ? dates[a].getTime() : -Infinity;
    const tEnd = dates[b] ? dates[b].getTime() + 3600_000 : Infinity;
    return report.episodes.filter((e) => {
      const t = new Date(e.ts_entry).getTime();
      return t >= tStart && t <= tEnd;
    });
  }, [report.episodes, dates, a, b]);

  const [sort, setSort] = useState<{ k: SortKey; dir: 'asc' | 'desc' }>({ k: 'net_pnl', dir: 'desc' });
  const [sel, setSel] = useState<Episode | null>(null);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState<FilterMode>('all');

  const filtered = useMemo(() => {
    let out = ep;
    if (filter === 'winners') out = out.filter((e) => e.net_pnl > 0);
    if (filter === 'losers') out = out.filter((e) => e.net_pnl < 0);
    if (filter === 'long') out = out.filter((e) => e.side > 0);
    if (filter === 'short') out = out.filter((e) => e.side < 0);
    if (search) {
      const q = search.toLowerCase();
      out = out.filter((e) => String(e.id).includes(q) || e.regime.includes(q));
    }
    out = [...out].sort((x, y) => {
      const d = (x[sort.k] as number) - (y[sort.k] as number);
      return sort.dir === 'asc' ? d : -d;
    });
    return out;
  }, [ep, filter, search, sort]);

  useEffect(() => {
    if (!filtered.length) { setSel(null); return; }
    if (!sel || !filtered.find((e) => e.id === sel.id)) setSel(filtered[0]);
  }, [filtered, sel]);

  function header(k: SortKey, label: string) {
    const cls = sort.k === k ? `sort-${sort.dir}` : '';
    return (
      <th className={cls}
        onClick={() => setSort((s) => s.k === k ? { k, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { k, dir: 'desc' })}>
        {label}
      </th>
    );
  }

  return (
    <div>
      <div className="tab-head">
        <span className="num">06</span>
        <h2>Trade Episodes</h2>
        <span className="desc">{filtered.length} of {ep.length} in range</span>
      </div>
      <HoldDurationPanel episodes={ep} />
      <div className="grid" style={{ gridTemplateColumns: '1.1fr 1fr' }}>
        <Panel
          title="Episodes"
          subtitle={
            <span style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input
                placeholder="id / regime"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                style={{
                  background: 'transparent', border: '1px solid var(--rule-2)',
                  color: 'var(--ink)', padding: '3px 8px', borderRadius: 2,
                  fontFamily: 'IBM Plex Mono', fontSize: 10, width: 90,
                }} />
              <span className="seg">
                {FILTERS.map((f) => (
                  <button key={f} className={filter === f ? 'on' : ''} onClick={() => setFilter(f)}>{f}</button>
                ))}
              </span>
            </span>
          }>
          <div className="scroll" style={{ maxHeight: 560 }}>
            <table className="etable">
              <thead>
                <tr>
                  {header('id', '#')}
                  <th>side</th>
                  {header('entry', 'entry')}
                  {header('dur', 'fires')}
                  {header('edge_entry', 'edge')}
                  {header('iv_entry', 'iv₀')}
                  {header('rv_realized', 'rv')}
                  {header('vega_pnl', 'vega')}
                  {header('theta_pnl', 'theta')}
                  {header('net_pnl', 'net$')}
                  <th>regime</th>
                </tr>
              </thead>
              <tbody>
                {filtered.slice(0, 300).map((e) => (
                  <tr key={e.id}
                    className={sel?.id === e.id ? 'selected' : ''}
                    onClick={() => setSel(e)}>
                    <td style={{ color: 'var(--ink)' }}>#{e.id}</td>
                    <td className={e.side > 0 ? 'side-l' : 'side-s'}>{e.side > 0 ? 'LONG' : 'SHORT'}</td>
                    <td>{e.ts_entry.slice(0, 10)}</td>
                    <td>{e.dur}</td>
                    <td className={e.edge_entry >= 0 ? 'pos' : 'neg'}>{e.edge_entry.toFixed(1)}</td>
                    <td>{e.iv_entry.toFixed(1)}</td>
                    <td>{e.rv_realized.toFixed(1)}</td>
                    <td className={e.vega_pnl >= 0 ? 'pos' : 'neg'}>{FMT.dol(e.vega_pnl, 0)}</td>
                    <td className={e.theta_pnl >= 0 ? 'pos' : 'neg'}>{FMT.dol(e.theta_pnl, 0)}</td>
                    <td className={e.net_pnl >= 0 ? 'pos' : 'neg'} style={{ fontWeight: 500 }}>
                      {FMT.dol(e.net_pnl, 0)}
                    </td>
                    <td style={{ color: 'var(--muted)' }}>{e.regime}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
        <div style={{ position: 'sticky', top: 112, alignSelf: 'start' }}>
          {sel && <EpisodeDetail ep={sel} />}
        </div>
      </div>
    </div>
  );
}

function HoldDurationPanel({ episodes }: { episodes: Episode[] }) {
  const buckets = useMemo(() => {
    const map = new Map<string, number>();
    const keys = ['1', '2', '3', '4', '5+'];
    for (const k of keys) map.set(k, 0);
    for (const e of episodes) {
      const k = e.dur >= 5 ? '5+' : String(e.dur);
      map.set(k, (map.get(k) || 0) + 1);
    }
    return keys.map((k) => ({ label: k, v: map.get(k) || 0 }));
  }, [episodes]);
  const total = episodes.length || 1;
  const pct1 = ((buckets[0].v / total) * 100).toFixed(0);
  return (
    <Panel
      title="Hold Duration Distribution"
      subtitle={`${pct1}% are single 15m fires · dur > 1 = consecutive same-direction fires (each still a 15m hold)`}
      tight>
      <BarChart bars={buckets} height={120}
        yFormat={(v) => v.toLocaleString()}
        colorMap={(b) => b.label === '1' ? 'var(--accent)' : 'var(--ink-2)'}
        tooltipLabel="fires" />
    </Panel>
  );
}

function Stat({ lbl, val }: { lbl: string; val: string }) {
  return (
    <div>
      <div style={{
        fontFamily: 'IBM Plex Mono', fontSize: 9, color: 'var(--muted)',
        textTransform: 'uppercase', letterSpacing: '0.08em',
      }}>{lbl}</div>
      <div style={{
        fontFamily: 'IBM Plex Mono', fontSize: 14, color: 'var(--ink)', marginTop: 2,
      }}>{val}</div>
    </div>
  );
}

function EpisodeDetail({ ep }: { ep: Episode }) {
  const report = useReport();
  const N = report.meta.n_bars;
  const pad = 12;
  const a = Math.max(0, ep.entry - pad);
  const b = Math.min(N - 1, ep.exit + pad);
  const px = report.bars_common.price.slice(a, b + 1);
  const iv = report.bars_common.iv_mkt.slice(a, b + 1);
  const ivModel = report.bars_common.iv_model.slice(a, b + 1);

  const localEntry = ep.entry - a;
  const localExit = ep.exit - a;
  const nb = Math.max(1, ep.exit - ep.entry);

  const cvega: number[] = [];
  const ctheta: number[] = [];
  const chedge: number[] = [];
  const ccost: number[] = [];
  let V = 0, T = 0, H = 0, C = 0;
  for (let i = 0; i < px.length; i++) {
    if (i >= localEntry && i < localEntry + nb) {
      T += ep.theta_pnl / nb;
      H += ep.delta_hedge_pnl / nb;
      C += -(ep.hedge_costs + ep.option_fees + ep.slippage) / nb;
    }
    if (i === localEntry + nb - 1) V += ep.vega_pnl;
    cvega.push(V); ctheta.push(T); chedge.push(H); ccost.push(C);
  }
  const legs = [
    { label: 'vega', color: '#14b8a6', y: cvega },
    { label: 'theta', color: '#8b5cf6', y: ctheta },
    { label: 'hedge', color: '#60a5fa', y: chedge },
    { label: 'costs', color: '#ef4444', y: ccost },
  ];

  return (
    <div className="panel">
      <div className="panel-hdr">
        <div>
          <div className="panel-title">
            Episode #{ep.id} · {ep.side > 0 ? 'LONG vol' : 'SHORT vol'} · {ep.regime}
          </div>
          <div style={{
            fontFamily: 'IBM Plex Mono', fontSize: 10, color: 'var(--muted)', marginTop: 2,
          }}>
            {ep.ts_entry} → {ep.ts_exit} · {ep.dur} × 15m fires (15m hold each) · {ep.N_contracts.toFixed(2)} contracts
          </div>
        </div>
        <div className="panel-sub">
          <span style={{ color: ep.net_pnl >= 0 ? 'var(--pos)' : 'var(--neg)', fontSize: 14 }}>
            {FMT.dol(ep.net_pnl, 0)}
          </span>
        </div>
      </div>
      <div className="panel-body tight">
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(4,1fr)',
          borderTop: '1px solid var(--rule)', borderBottom: '1px solid var(--rule)',
          margin: '4px -10px 10px', padding: '6px 10px',
        }}>
          <Stat lbl="entry edge" val={ep.edge_entry.toFixed(1)} />
          <Stat lbl="iv₀ → iv₁" val={`${ep.iv_entry.toFixed(1)} → ${ep.iv_exit.toFixed(1)}`} />
          <Stat lbl="rv (realized)" val={ep.rv_realized.toFixed(1)} />
          <Stat lbl="btc move" val={FMT.signedPct(ep.ret_btc, 1)} />
        </div>
        <div style={{
          fontFamily: 'IBM Plex Mono', fontSize: 10, color: 'var(--muted)', margin: '0 0 4px',
        }}>price context (± {pad} bars)</div>
        <LineChart height={120}
          series={[{ label: 'btc', y: px, color: 'var(--ink-2)', width: 1 }]}
          yFormat={(v) => `$${(v / 1000).toFixed(1)}k`}
          annotations={[{ i: localEntry, label: 'entry' }, { i: localExit, label: 'exit' }]}
          xLabel={(i) => String(i - localEntry)}
          crosshair={false} />
        <div style={{
          fontFamily: 'IBM Plex Mono', fontSize: 10, color: 'var(--muted)', margin: '6px 0 4px',
        }}>iv market vs iv model</div>
        <LineChart height={110}
          series={[
            { label: 'iv mkt', y: iv, color: 'var(--accent)' },
            { label: 'iv model', y: ivModel, color: 'var(--accent-2)' },
          ]}
          yFormat={(v) => v.toFixed(0)}
          annotations={[{ i: localEntry, label: 'in' }, { i: localExit, label: 'out' }]}
          xLabel={(i) => String(i - localEntry)}
          crosshair={false} />
        <div style={{
          fontFamily: 'IBM Plex Mono', fontSize: 10, color: 'var(--muted)', margin: '6px 0 4px',
        }}>cumulative legs ($)</div>
        <StackedArea xLen={legs[0].y.length} layers={legs} height={140}
          padding={{ t: 10, r: 12, b: 18, l: 52 }}
          yFormat={(v) => FMT.dol(v, 0)}
          xLabel={(i) => String(i - localEntry)} />
      </div>
    </div>
  );
}
