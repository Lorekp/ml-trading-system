import { useEffect, useState } from 'react';
import { ReportProvider, useReportStatus } from './data/ReportProvider';
import { CrosshairProvider } from './charts/CrosshairContext';
import { ControlsBar } from './controls/ControlsBar';
import { TabOverview } from './tabs/TabOverview';
import { TabAttribution } from './tabs/TabAttribution';
import { TabEdge } from './tabs/TabEdge';
import { TabRobustness } from './tabs/TabRobustness';
import { TabRegime } from './tabs/TabRegime';
import { TabEpisodes } from './tabs/TabEpisodes';
import { TweaksPanel, useTweakHostMessages } from './tweaks/TweaksPanel';
import { useTabKeyboard, type TabKey } from './lib/keyboard';

const TABS: { key: TabKey; num: string; label: string; cmp: () => JSX.Element }[] = [
  { key: '1', num: '01', label: 'Overview', cmp: TabOverview },
  { key: '2', num: '02', label: 'Attribution', cmp: TabAttribution },
  { key: '3', num: '03', label: 'Edge', cmp: TabEdge },
  { key: '4', num: '04', label: 'Robustness', cmp: TabRobustness },
  { key: '5', num: '05', label: 'Regime', cmp: TabRegime },
  { key: '6', num: '06', label: 'Episodes', cmp: TabEpisodes },
];

function Header() {
  return (
    <header className="top">
      <div>
        <h1>Backtest Dashboard <span className="dim">— BTC vol-arb v2</span></h1>
        <div className="sub">compound equity · 1h cadence</div>
      </div>
      <div className="meta">
        15-min source · downsampled<br />
        <span className="pill">v2 production</span>
      </div>
    </header>
  );
}

function TabBar({ active, setActive }: { active: TabKey; setActive: (k: TabKey) => void }) {
  return (
    <nav className="tabs">
      {TABS.map((t) => (
        <button key={t.key} className={active === t.key ? 'on' : ''} onClick={() => setActive(t.key)}>
          <span className="num">{t.num}</span>{t.label}
        </button>
      ))}
    </nav>
  );
}

function Loading({ error }: { error: string | null }) {
  return (
    <div className="loading">
      {error
        ? <span style={{ color: 'var(--neg)' }}>error loading report.json: {error}</span>
        : <><span className="pulse" />loading report.json…</>}
    </div>
  );
}

function Body() {
  const { report, loading, error } = useReportStatus();
  const [active, setActive] = useState<TabKey>(() => {
    const saved = localStorage.getItem('bt.tab');
    return (saved && (TABS as readonly { key: string }[]).some((t) => t.key === saved)) ? saved as TabKey : '1';
  });
  const [tweaksOpen, setTweaksOpen] = useState(false);

  useTabKeyboard({ active, setActive });
  useTweakHostMessages(setTweaksOpen);

  useEffect(() => { localStorage.setItem('bt.tab', active); }, [active]);

  if (loading || !report) return <><Header /><Loading error={error} /></>;

  const Tab = TABS.find((t) => t.key === active)!.cmp;
  return (
    <>
      <Header />
      <ControlsBar />
      <TabBar active={active} setActive={setActive} />
      <main><Tab /></main>
      <TweaksPanel open={tweaksOpen} onClose={() => setTweaksOpen(false)} />
    </>
  );
}

export default function App() {
  return (
    <ReportProvider>
      <CrosshairProvider>
        <Body />
      </CrosshairProvider>
    </ReportProvider>
  );
}
