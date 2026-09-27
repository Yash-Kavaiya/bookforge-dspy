/** Display formatting shared across the build/progress views. Pure functions
 *  only — no Angular pipes, so they're trivially usable from both templates
 *  (via component methods) and services building strings for a live log. */

export function fmtBytes(n: number): string {
  if (!n) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(Math.floor(Math.log(n) / Math.log(1024)), units.length - 1);
  return `${(n / 1024 ** i).toFixed(i ? 1 : 0)} ${units[i]}`;
}

export function fmtDuration(s: number | null | undefined): string {
  if (!s) return '';
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : `${m}m`;
}

export function fmtTokens(n: number): string {
  if (!n) return '0';
  if (n >= 1000) return `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k`;
  return String(n);
}

// A fixed-width 24h clock avoids locale-dependent AM/PM strings that don't
// fit the log's timestamp column and wrap onto a second line.
export function fmtClock(iso: string): string {
  if (!iso) return '';
  const d = new Date(iso);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export function fmtCost(usd: number): string {
  if (usd === 0) return '$0';
  return usd < 0.01 ? `$${usd.toFixed(4)}` : `$${usd.toFixed(2)}`;
}

// USD per 1,000,000 tokens — approximate public list prices, not a billing
// API. Matched by exact model id first, then by prefix so an NVIDIA-NIM-style
// "openai/nvidia/..." id doesn't collide with plain "openai/gpt-4o".
const PRICING: Record<string, [number, number]> = {
  'openai/gpt-4o': [2.5, 10],
  'openai/gpt-4o-mini': [0.15, 0.6],
  'openai/gpt-4.1': [2.0, 8.0],
  'openai/gpt-4.1-mini': [0.4, 1.6],
  'openai/gpt-4.1-nano': [0.1, 0.4],
  'openai/o4-mini': [1.1, 4.4],
  'gemini/gemini-2.5-pro': [1.25, 5.0],
  'gemini/gemini-2.5-flash': [0.3, 2.5],
  'gemini/gemini-2.0-flash': [0.1, 0.4],
  'anthropic/claude-sonnet-4-5': [3.0, 15.0],
  'anthropic/claude-opus-4-5': [15.0, 75.0],
  'anthropic/claude-haiku-4-5': [1.0, 5.0],
};

export function priceFor(model: string | undefined): [number, number] | null {
  if (!model) return null;
  if (PRICING[model]) return PRICING[model];
  const prefix = Object.keys(PRICING).find((k) => model.startsWith(k));
  return prefix ? PRICING[prefix] : null;
}
