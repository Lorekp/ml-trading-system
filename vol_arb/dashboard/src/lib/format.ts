// Number formatters mirroring the prototype's FMT.* helpers.

export const FMT = {
  pct: (x: number, d = 2) => `${(x * 100).toFixed(d)}%`,
  bps: (x: number, d = 0) => `${(x * 10000).toFixed(d)} bps`,
  dol: (x: number, d = 0) => {
    const sign = x < 0 ? '-$' : '$';
    return sign + Math.abs(x).toLocaleString(undefined, { maximumFractionDigits: d });
  },
  num: (x: number, d = 2) => x.toFixed(d),
  ratio: (x: number, d = 2) => x.toFixed(d),
  signed: (x: number, d = 2) => (x >= 0 ? '+' : '') + x.toFixed(d),
  signedPct: (x: number, d = 2) => (x >= 0 ? '+' : '') + (x * 100).toFixed(d) + '%',
};

export function niceTicks(lo: number, hi: number, n = 5): number[] {
  const span = hi - lo;
  if (span === 0) return [lo];
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  let step: number;
  if (norm < 1.5) step = 1 * mag;
  else if (norm < 3) step = 2 * mag;
  else if (norm < 7) step = 5 * mag;
  else step = 10 * mag;
  const out: number[] = [];
  const start = Math.ceil(lo / step) * step;
  for (let v = start; v <= hi + 1e-9; v += step) out.push(+v.toFixed(12));
  return out;
}

export function fmtDate(d: Date): string {
  return d.toISOString().slice(0, 10);
}

export function fmtDateTime(d: Date): string {
  return d.toISOString().slice(0, 16).replace('T', ' ');
}
