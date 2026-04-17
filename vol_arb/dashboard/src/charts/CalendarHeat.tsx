import { Heatmap } from './Heatmap';
import type { MonthlyReturn } from '../types';

interface CalendarHeatProps {
  monthData: MonthlyReturn[];
  height?: number;
  fmt?: (v: number) => string;
  domain?: number | null;
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

export function CalendarHeat({
  monthData, height = 150,
  fmt = (v) => `${(v * 100).toFixed(1)}%`,
  domain = null,
}: CalendarHeatProps) {
  const years = Array.from(new Set(monthData.map((d) => d.k.slice(0, 4)))).sort();
  const data = years.map((y) =>
    MONTHS.map((_, mi) => {
      const k = `${y}-${String(mi + 1).padStart(2, '0')}`;
      const row = monthData.find((d) => d.k === k);
      return row ? row.r : null;
    })
  );
  return <Heatmap rows={years} cols={MONTHS} data={data} height={height} fmt={fmt} domain={domain} />;
}
