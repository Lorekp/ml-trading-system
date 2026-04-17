// Schema mirrors etl/build_report.py output exactly.

export interface BarsCommon {
  ts: string[];           // ISO UTC
  price: number[];
  iv_mkt: number[];
  iv_model: number[];
  edge: number[];
  signals: number[];      // -1 / 0 / 1
}

export interface BarsByMode {
  equity: number[];
  equity_btc_denom: number[];  // idle capital held in BTC instead of USD cash
  vega_pnl_bar: number[];
  theta_pnl_bar: number[];
  delta_hedge_pnl_bar: number[];
  costs_bar: number[];    // already negated to be ≤0
}

export interface Episode {
  id: number;
  entry: number;
  exit: number;
  side: -1 | 1;
  dur: number;
  ts_entry: string;
  ts_exit: string;
  iv_entry: number;
  iv_exit: number;
  rv_entry: number;
  rv_realized: number;
  edge_entry: number;
  vrp: number;
  vega_pnl: number;
  theta_pnl: number;
  delta_hedge_pnl: number;
  hedge_costs: number;
  option_fees: number;
  slippage: number;
  gross_pnl: number;
  net_pnl: number;
  N_contracts: number;
  regime: 'rich' | 'cheap' | 'neutral' | 'stress';
  price_entry: number;
  price_exit: number;
  ret_btc: number;
}

export interface HeadlineStats {
  cagr: number;
  vol: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  maxdd: number;
  maxdd_dur_days: number;
  hit: number;
  n_obs: number;
  sharpe_ci: [number, number];
  total_ret: number;
}

export interface DecileRow {
  d: number;
  mean: number;
  se: number;
  n: number;
  edge_lo: number;
  edge_hi: number;
}

export interface MonthlyReturn { k: string; r: number; }
export interface RollingICPoint { i: number; ic: number; }
export interface StressEvent { name: string; date: string; pnl: number; bench: number; }
export interface CostSensitivityCell { thr: number; cost: number; sharpe: number; }
export interface WalkForwardFold {
  i: number;
  ts_a: string;
  ts_b: string;
  sharpe: number;
  cagr: number;
  dd: number;
  is_: boolean;
  oos: boolean;
}

export interface BacktestReport {
  meta: {
    strategy: string;
    subject: string;
    n_bars: number;
    bar_hours: number;
    start: string;
    end: string;
    generated_at: string;
    E0: number;
  };
  bars_common: BarsCommon;
  bars_nc: BarsByMode;
  bars_comp: BarsByMode;
  episodes: Episode[];
  bench_6040: number[];
  bench_btc_hold: number[];
  bench_cash: number[];
  decile_table: DecileRow[];
  monthly_strat: MonthlyReturn[];
  monthly_bench: MonthlyReturn[];
  rolling_ic: RollingICPoint[];
  rolling_sharpe: (number | null)[];
  drawdown_nc: number[];
  drawdown_comp: number[];
  stress_events: StressEvent[];
  cost_sensitivity: CostSensitivityCell[];
  bootstrap_sharpe: number[];
  permutation_null: number[];
  permutation_obs: number;
  walk_forward: WalkForwardFold[];
  walk_forward_summary: {
    mean_fold_sharpe: number;
    mean_oos_fold_sharpe: number;
    full_oos_sharpe: number;
  };
  fire_rate_by_year: { year: number; n_bars: number; n_starts: number; rate: number }[];
  daily_volume: { date: string; n_contracts: number; n_episodes: number }[];
  split_pnl: {
    oos_start: string;
    is_bars: number;
    oos_bars: number;
    is_net_pnl: number;
    oos_net_pnl: number;
  };
  headline_stats_nc: HeadlineStats;
  headline_stats_comp: HeadlineStats;
}

export type Mode = 'nc' | 'comp';
export type Benchmark = '60/40' | 'BTC hold' | 'Cash';
export type Freq = '1h' | '1d' | '1w';
export type SignalMode = 'All' | 'Long only' | 'Short only';

export interface GlobalControls {
  range: [number, number];
  benchmark: Benchmark;
  freq: Freq;
  costs: number;
  signalMode: SignalMode;
  mode: Mode;
}

export interface TweakState {
  numfmt: 'bps' | 'pct' | 'dollar';
  decimals: 1 | 2 | 3;
  chartstyle: 'line' | 'area' | 'step';
  bands: 'on' | 'off';
}
