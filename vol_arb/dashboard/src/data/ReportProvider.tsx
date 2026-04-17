import { createContext, useContext, useEffect, useMemo, useState, ReactNode } from 'react';
import type { BacktestReport, GlobalControls, Mode } from '../types';

interface ReportCtx {
  report: BacktestReport | null;
  loading: boolean;
  error: string | null;
}

const ReportContext = createContext<ReportCtx>({ report: null, loading: true, error: null });

interface ControlsCtx {
  controls: GlobalControls;
  setControls: (updater: (c: GlobalControls) => GlobalControls) => void;
}

const ControlsContext = createContext<ControlsCtx | null>(null);

const DEFAULT_CONTROLS: Omit<GlobalControls, 'range'> = {
  benchmark: '60/40',
  freq: '1h',
  costs: 1.0,
  signalMode: 'All',
  mode: 'comp',
};

export function ReportProvider({ children }: { children: ReactNode }) {
  const [report, setReport] = useState<BacktestReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [controls, setControlsState] = useState<GlobalControls | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch('/report.json', { cache: 'no-cache' })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: BacktestReport) => {
        if (cancelled) return;
        setReport(data);
        setControlsState({
          ...DEFAULT_CONTROLS,
          range: [0, data.meta.n_bars - 1],
        });
      })
      .catch((e) => {
        if (cancelled) return;
        setError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const setControls = (updater: (c: GlobalControls) => GlobalControls) => {
    setControlsState((c) => (c ? updater(c) : c));
  };

  const reportCtx = useMemo<ReportCtx>(
    () => ({ report, loading: !report && !error, error }),
    [report, error]
  );

  return (
    <ReportContext.Provider value={reportCtx}>
      <ControlsContext.Provider value={controls ? { controls, setControls } : null}>
        {children}
      </ControlsContext.Provider>
    </ReportContext.Provider>
  );
}

export function useReport(): BacktestReport {
  const ctx = useContext(ReportContext);
  if (!ctx.report) throw new Error('Report not loaded');
  return ctx.report;
}

export function useReportStatus() {
  return useContext(ReportContext);
}

export function useControls(): GlobalControls {
  const ctx = useContext(ControlsContext);
  if (!ctx) throw new Error('Controls not initialized');
  return ctx.controls;
}

export function useSetControls() {
  const ctx = useContext(ControlsContext);
  if (!ctx) throw new Error('Controls not initialized');
  return ctx.setControls;
}

// Selectors

export function useBars(mode?: Mode) {
  const report = useReport();
  const controls = useControls();
  const m = mode ?? controls.mode;
  return m === 'nc' ? report.bars_nc : report.bars_comp;
}

export function useDrawdown(mode?: Mode) {
  const report = useReport();
  const controls = useControls();
  const m = mode ?? controls.mode;
  return m === 'nc' ? report.drawdown_nc : report.drawdown_comp;
}

export function useHeadlineStats(mode?: Mode) {
  const report = useReport();
  const controls = useControls();
  const m = mode ?? controls.mode;
  return m === 'nc' ? report.headline_stats_nc : report.headline_stats_comp;
}

export function useBenchmarkSeries() {
  const report = useReport();
  const { benchmark } = useControls();
  switch (benchmark) {
    case 'BTC hold': return report.bench_btc_hold;
    case 'Cash':     return report.bench_cash;
    case '60/40':
    default:         return report.bench_6040;
  }
}

// Pre-parse ts strings into Date objects once.
export function useDates(): Date[] {
  const report = useReport();
  return useMemo(() => report.bars_common.ts.map((t) => new Date(t)), [report]);
}
