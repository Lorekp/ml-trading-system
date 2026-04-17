import { useReport, useControls, useSetControls, useDates } from '../data/ReportProvider';
import { DateRangeBrush } from './DateRangeBrush';
import { SegControl } from './SegControl';
import type { Benchmark, Freq, SignalMode, Mode } from '../types';

const BENCHMARKS: readonly Benchmark[] = ['60/40', 'BTC hold', 'Cash'];
const FREQS: readonly Freq[] = ['1h', '1d', '1w'];
const SIGNAL_MODES: readonly SignalMode[] = ['All', 'Long only', 'Short only'];
const MODES: readonly Mode[] = ['nc', 'comp'];

export function ControlsBar() {
  const report = useReport();
  const controls = useControls();
  const setControls = useSetControls();
  const dates = useDates();
  const N = report.meta.n_bars;

  const presets = [
    { label: 'All', range: [0, N - 1] as [number, number] },
    {
      label: 'YTD',
      range: (() => {
        const target = new Date(`${dates[N - 1].getUTCFullYear()}-01-01T00:00:00Z`);
        for (let i = 0; i < N; i++) if (dates[i] >= target) return [i, N - 1] as [number, number];
        return [0, N - 1] as [number, number];
      })(),
    },
    { label: '1Y', range: [Math.max(0, N - 365 * 24), N - 1] as [number, number] },
    { label: '6M', range: [Math.max(0, N - 180 * 24), N - 1] as [number, number] },
    { label: '3M', range: [Math.max(0, N - 90 * 24), N - 1] as [number, number] },
  ];

  const fmtD = (i: number) => dates[i].toISOString().slice(0, 10);

  return (
    <div className="ctrls">
      <div className="c">
        <span className="lbl">Date range</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <DateRangeBrush
            value={controls.range}
            onChange={(v) => setControls((c) => ({ ...c, range: v }))} />
          <span className="daterange-label">{fmtD(controls.range[0])} → {fmtD(controls.range[1])}</span>
        </div>
      </div>
      <div className="c">
        <span className="lbl">Presets</span>
        <span className="seg">
          {presets.map((p) => (
            <button key={p.label} onClick={() => setControls((c) => ({ ...c, range: p.range }))}>
              {p.label}
            </button>
          ))}
        </span>
      </div>
      <div className="c">
        <span className="lbl">Benchmark</span>
        <SegControl options={BENCHMARKS} value={controls.benchmark}
          onChange={(v) => setControls((c) => ({ ...c, benchmark: v }))} />
      </div>
      <div className="c">
        <span className="lbl">Sample freq</span>
        <SegControl options={FREQS} value={controls.freq}
          onChange={(v) => setControls((c) => ({ ...c, freq: v }))} />
      </div>
      <div className="c">
        <span className="lbl">Costs × {controls.costs.toFixed(1)}</span>
        <input type="range" min={0} max={3} step={0.1} value={controls.costs}
          onChange={(e) => setControls((c) => ({ ...c, costs: +e.target.value }))} />
      </div>
      <div className="c">
        <span className="lbl">Signal set</span>
        <SegControl options={SIGNAL_MODES} value={controls.signalMode}
          onChange={(v) => setControls((c) => ({ ...c, signalMode: v }))} />
      </div>
      <div className="c" style={{ marginLeft: 'auto' }}>
        <span className="lbl">Mode</span>
        <SegControl options={MODES} value={controls.mode}
          labels={{ nc: 'NC', comp: 'Comp' }}
          onChange={(v) => setControls((c) => ({ ...c, mode: v }))} />
      </div>
    </div>
  );
}
