interface StatRow {
  k: string;
  v: string;
  ci?: string;
}

export function StatTable({ rows }: { rows: StatRow[] }) {
  return (
    <div className="stat-table">
      {rows.map((r, i) => (
        <div className="row" key={i}>
          <div className="k">{r.k}</div>
          <div className="v">{r.v}</div>
          {r.ci && <div className="ci">{r.ci}</div>}
        </div>
      ))}
    </div>
  );
}
