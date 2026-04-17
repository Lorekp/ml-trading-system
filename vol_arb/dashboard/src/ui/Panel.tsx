import { ReactNode } from 'react';

interface PanelProps {
  title: string;
  subtitle?: ReactNode;
  tight?: boolean;
  children: ReactNode;
  style?: React.CSSProperties;
}

export function Panel({ title, subtitle, tight, children, style }: PanelProps) {
  return (
    <div className="panel" style={style}>
      <div className="panel-hdr">
        <div className="panel-title">{title}</div>
        {subtitle != null && <div className="panel-sub">{subtitle}</div>}
      </div>
      <div className={`panel-body${tight ? ' tight' : ''}`}>{children}</div>
    </div>
  );
}

interface ChipProps {
  color: string;
  dashed?: boolean;
  label: string;
}

export function Chip({ color, dashed, label }: ChipProps) {
  return (
    <span className="chip">
      <span
        className={dashed ? 'sw dash' : 'sw'}
        style={dashed ? { color } : { background: color }}
      />
      {label}
    </span>
  );
}
