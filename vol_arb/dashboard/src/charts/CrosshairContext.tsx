import { createContext, useContext, useState, ReactNode } from 'react';

interface CrosshairCtx {
  cursor: number | null;
  setCursor: (idx: number | null) => void;
}

const ctx = createContext<CrosshairCtx>({ cursor: null, setCursor: () => {} });

export function CrosshairProvider({ children }: { children: ReactNode }) {
  const [cursor, setCursor] = useState<number | null>(null);
  return <ctx.Provider value={{ cursor, setCursor }}>{children}</ctx.Provider>;
}

export function useCrosshair() {
  return useContext(ctx);
}
