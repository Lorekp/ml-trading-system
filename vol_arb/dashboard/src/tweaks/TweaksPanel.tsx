import { useEffect, useState } from 'react';
import type { TweakState } from '../types';

const TWEAK_DEFAULTS: TweakState = {
  numfmt: 'pct',
  decimals: 2,
  chartstyle: 'line',
  bands: 'on',
};

interface TweaksPanelProps {
  open: boolean;
  onClose: () => void;
}

export function TweaksPanel({ open, onClose }: TweaksPanelProps) {
  const [tw, setTw] = useState<TweakState>(TWEAK_DEFAULTS);

  useEffect(() => {
    function onMsg(e: MessageEvent) {
      const d = e.data ?? {};
      if (d.type === '__activate_edit_mode' || d.type === '__deactivate_edit_mode') {
        // host-driven; parent App handles open state
      }
    }
    window.addEventListener('message', onMsg);
    try {
      window.parent.postMessage({ type: '__edit_mode_available' }, '*');
    } catch { /* not in iframe */ }
    return () => window.removeEventListener('message', onMsg);
  }, []);

  function set<K extends keyof TweakState>(k: K, v: TweakState[K]) {
    const next = { ...tw, [k]: v };
    setTw(next);
    try {
      window.parent.postMessage({ type: '__edit_mode_set_keys', edits: { [k]: v } }, '*');
    } catch { /* ignore */ }
  }

  if (!open) return null;
  return (
    <div className="tweaks on">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h4>Tweaks</h4>
        <button onClick={onClose}
          style={{
            background: 'transparent', border: 0, color: 'var(--muted)',
            fontFamily: 'IBM Plex Mono', fontSize: 11,
          }}>×</button>
      </div>
      <div className="t-row">
        <label>Number format</label>
        <span className="seg">
          {(['bps', 'pct', 'dollar'] as const).map((v) => (
            <button key={v} className={tw.numfmt === v ? 'on' : ''} onClick={() => set('numfmt', v)}>{v}</button>
          ))}
        </span>
      </div>
      <div className="t-row">
        <label>Decimals</label>
        <span className="seg">
          {([1, 2, 3] as const).map((v) => (
            <button key={v} className={tw.decimals === v ? 'on' : ''} onClick={() => set('decimals', v)}>{v}</button>
          ))}
        </span>
      </div>
      <div className="t-row">
        <label>Chart style</label>
        <span className="seg">
          {(['line', 'area', 'step'] as const).map((v) => (
            <button key={v} className={tw.chartstyle === v ? 'on' : ''} onClick={() => set('chartstyle', v)}>{v}</button>
          ))}
        </span>
      </div>
      <div className="t-row">
        <label>Confidence bands</label>
        <span className="seg">
          {(['on', 'off'] as const).map((v) => (
            <button key={v} className={tw.bands === v ? 'on' : ''} onClick={() => set('bands', v)}>{v}</button>
          ))}
        </span>
      </div>
    </div>
  );
}

export function useTweakHostMessages(setOpen: (v: boolean) => void) {
  useEffect(() => {
    function onMsg(e: MessageEvent) {
      const d = e.data ?? {};
      if (d.type === '__activate_edit_mode') setOpen(true);
      if (d.type === '__deactivate_edit_mode') setOpen(false);
    }
    window.addEventListener('message', onMsg);
    return () => window.removeEventListener('message', onMsg);
  }, [setOpen]);
}
