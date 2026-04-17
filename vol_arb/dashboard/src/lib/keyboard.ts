import { useEffect } from 'react';

export const TAB_KEYS = ['1', '2', '3', '4', '5', '6'] as const;
export type TabKey = typeof TAB_KEYS[number];

interface KeyboardArgs {
  active: TabKey;
  setActive: (k: TabKey) => void;
}

export function useTabKeyboard({ active, setActive }: KeyboardArgs) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
      if ((TAB_KEYS as readonly string[]).includes(e.key)) {
        setActive(e.key as TabKey);
        return;
      }
      const i = TAB_KEYS.indexOf(active);
      if (e.key === 'ArrowRight') setActive(TAB_KEYS[(i + 1) % TAB_KEYS.length]);
      if (e.key === 'ArrowLeft') setActive(TAB_KEYS[(i - 1 + TAB_KEYS.length) % TAB_KEYS.length]);
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [active, setActive]);
}
