// Direct DOM tooltip — uses the global #tip element so all charts share one tooltip surface.

let tipEl: HTMLDivElement | null = null;

function getTip(): HTMLDivElement {
  if (!tipEl) tipEl = document.getElementById('tip') as HTMLDivElement;
  return tipEl;
}

export function showTip(html: string, e: { clientX: number; clientY: number }) {
  const t = getTip();
  if (!t) return;
  t.innerHTML = html;
  t.style.display = 'block';
  const r = t.getBoundingClientRect();
  const x = e.clientX + 14;
  const y = e.clientY + 14;
  const maxX = window.innerWidth - r.width - 8;
  t.style.left = Math.min(x, maxX) + 'px';
  t.style.top = Math.min(y, window.innerHeight - r.height - 8) + 'px';
}

export function hideTip() {
  const t = getTip();
  if (t) t.style.display = 'none';
}

export function tooltipRow(color: string, label: string, value: string): string {
  return `<div><span class="tk" style="color:${color}">● ${label}</span><span class="tv">${value}</span></div>`;
}

export function tooltipHeader(text: string): string {
  return `<div class="th">${text}</div>`;
}
