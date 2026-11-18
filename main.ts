// gta6-twitch-ccu-forecast-01 entry (Node 22, standard library only).
//
// Forecast: an ensemble of past big launch days, expressed as log-ratios to
// each launch's hour-1 print, with the shape adjusted for how big hour 1 was.
//
// What the data says (38 past launches in every practice case, 42 in the
// final; one is the GTA VI category's own pre-release history, which has no
// release day and drops out by itself):
//   - A launch's release day has a strong diurnal shape: it dips to ~0.75x
//     hour 1 overnight (hours 3-11 UTC), then climbs as the US wakes and
//     unlocks (US East at hour 5, US West at hour 8) to a peak of ~1.5x hour 1
//     in US-afternoon / EU-evening hours 19-22, then decays.
//   - Not every comparable is the same kind of launch. Games whose hour 1 is
//     a tiny fraction of their day peak (v1/peak < 0.15: Silksong, Black Myth,
//     Baldur's Gate 3, ...) unlocked later in the day on PC (one global
//     unlock hour), so their "shape" is really the unlock clock, not the
//     diurnal curve. GTA VI is console-only with a rolling midnight unlock,
//     so only launches with a genuine midnight hour 1 (v1/peak >= 0.15)
//     enter the pool. That leaves 26 of 37 in the practice cases.
//   - The shape depends on the size of hour 1, strongly: across the pool the
//     correlation of log(hour-1 viewers) with the log-ratio at the evening
//     peak is about -0.7. A launch already huge at midnight (Elden Ring,
//     Cyberpunk: hour 1 IS near the peak) stays flat; a small midnight print
//     grows several-fold through the day. So per forecast hour we regress the
//     log-ratio on log(hour-1 viewers) across the pool and evaluate the fit
//     at this case's hour-1 print (clamped to just outside the pool's range
//     so a record-breaking GTA VI hour 1 cannot extrapolate wildly).
//   - Spread: the residual standard deviation of that regression, per hour,
//     with Student-t (5 df) quantiles instead of normal ones. Launch shapes
//     are heavy-tailed (one small launch peaked 9x its hour 1); the t tails
//     buy honest 90% coverage without widening the middle, where most of the
//     pinball mass sits.
//
// Validation (leave-one-out over the 26 pool launches, plus the 4 true
// practice answers fetched from SullyGnome's hourly averages):
//   pinball loss on the log scale, mean over hours 2-24 x 5 quantiles:
//     house baseline (starter):  LOO 0.284, practice 0.387
//     this forecaster:           LOO 0.197, practice 0.249
//   typical miss: baseline 58% / 90% (LOO / practice); this: 40% / 53%.
//   90% coverage: baseline 0.82 / 0.78; this: 0.89 / 0.89.
// The practice number for the baseline (0.387) reproduces the figure the
// rules quote, which also confirms the fetched practice answers are the
// right ones. This forecaster beats the baseline on all four practice
// cases individually (Wolverine 0.42 vs 0.74, RDR2 0.24 vs 0.29,
// God of War 0.15 vs 0.21, Spider-Man 2 0.19 vs 0.30).
//
// Runs offline, deterministic, no randomness, no network. Reads every *.in
// in $MAKEFUN_INPUT_DIR (else /input) in name order and writes one
// <id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Degrees of freedom of the Student-t used for the forecast quantiles. */
const T_DF = 5;
/** A comparable joins the pool only when its hour 1 is at least this share of its day peak. */
const MIN_V1_OVER_PEAK = 0.15;
/** Floor on the pool size; if the filter leaves too few, it is relaxed. */
const MIN_POOL = 8;
/** How far (in log viewers) the case's hour-1 print may sit outside the pool's range. */
const CLAMP_LOW = 2.0;
const CLAMP_HIGH = 0.5;

interface HourlyPoint {
  from: string;
  viewers: number;
}

interface Comparable {
  day_start: string;
  hourly: HourlyPoint[];
}

interface CaseFile {
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  history_hourly?: Array<{ viewers: number }>;
  comparables: Comparable[];
}

/** The standard normal's quantile at p, Acklam's rational approximation. */
const normalQuantile = (p: number): number => {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const plow = 0.02425;
  const phigh = 1 - plow;
  let q: number;
  let r: number;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > phigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
};

/** Student-t quantile at p with `df` degrees of freedom (Cornish-Fisher expansion). */
const tQuantile = (p: number, df: number): number => {
  const z = normalQuantile(p);
  const g1 = (z * z * z + z) / 4;
  const g2 = (5 * z * z * z * z * z + 16 * z * z * z + 3 * z) / 96;
  return z + g1 / df + g2 / (df * df);
};

/** A comparable's 24 release-day viewer counts, or null when the day is incomplete. */
const releaseDay = (comp: Comparable): number[] | null => {
  const day = comp.day_start.slice(0, 10);
  const hours = comp.hourly
    .filter((h) => h.from.slice(0, 10) === day)
    .sort((a, b) => (a.from < b.from ? -1 : a.from > b.from ? 1 : 0));
  if (hours.length !== 24) return null;
  if (!hours[0].from.endsWith('T00:00:00Z') && !hours[0].from.startsWith(`${day}T00:`)) return null;
  const v = hours.map((h) => Math.max(1, Math.floor(h.viewers) || 1));
  return v;
};

/** The five quantiles of hour `h` for a case whose hour-1 print is `p1`. */
const hourQuantiles = (pool: number[][], p1: number, h: number): number[] => {
  // log(hour-1 viewers) and log-ratios of this hour across the pool.
  const xs = pool.map((v) => Math.log(v[0]));
  const rs = pool.map((v) => Math.log(v[h - 1] / v[0]));
  const n = xs.length;
  const mx = xs.reduce((s, x) => s + x, 0) / n;
  const my = rs.reduce((s, r) => s + r, 0) / n;
  let vxx = 0;
  let vxy = 0;
  for (let i = 0; i < n; i++) {
    vxx += (xs[i] - mx) * (xs[i] - mx);
    vxy += (xs[i] - mx) * (rs[i] - my);
  }
  const slope = vxx > 0 ? vxy / vxx : 0;
  // This case's hour-1 size, clamped just outside the pool's observed range.
  const lo = Math.min(...xs) - CLAMP_LOW;
  const hi = Math.max(...xs) + CLAMP_HIGH;
  const lv = Math.min(Math.max(Math.log(p1), lo), hi);
  const mu = my + slope * (lv - mx);
  // Residual spread of the regression (unbiased, n - 2 divisor).
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const fit = my + slope * (xs[i] - mx);
    ss += (rs[i] - fit) * (rs[i] - fit);
  }
  const sd = Math.max(Math.sqrt(ss / Math.max(n - 2, 1)), 0.05);
  const qs = LEVELS.map((tau) => p1 * Math.exp(mu + tQuantile(tau, T_DF) * sd));
  // Guardrails: finite, above 0, at most 1e9, non-decreasing, 2 decimals.
  const out: number[] = [];
  for (const q of qs) {
    let v = Number.isFinite(q) ? q : p1;
    v = Math.min(Math.max(v, 1), 1e9);
    v = Math.round(v * 100) / 100;
    if (out.length > 0 && v < out[out.length - 1]) v = out[out.length - 1];
    out.push(v);
  }
  return out;
};

/** The house baseline's persistence forecast, used only if no comparable qualifies. */
const baselineQuantiles = (p1: number, h: number): number[] => {
  const z = [-1.645, -0.674, 0, 0.674, 1.645];
  const out: number[] = [];
  for (const zi of z) {
    let v = Math.round(p1 * Math.exp(zi * 0.15 * Math.sqrt(h - 1)) * 100) / 100;
    v = Math.min(Math.max(v, 1), 1e9);
    if (out.length > 0 && v < out[out.length - 1]) v = out[out.length - 1];
    out.push(v);
  }
  return out;
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const first = c.known.find((k) => k.hour === 1);
  let p1 = first === undefined ? NaN : first.viewers;
  if (!Number.isFinite(p1) || p1 < 1) {
    // No usable hour-1 print: fall back to the last pre-checkpoint hourly point.
    const hist = c.history_hourly ?? [];
    p1 = hist.length > 0 ? hist[hist.length - 1].viewers : 1;
  }
  p1 = Math.max(1, p1);

  // The pool: comparables with a complete 24-hour release day whose hour 1
  // is a genuine midnight print (at least 15% of the day's peak).
  const complete: number[][] = [];
  for (const comp of c.comparables ?? []) {
    const v = releaseDay(comp);
    if (v !== null) complete.push(v);
  }
  let pool = complete.filter((v) => v[0] / Math.max(...v) >= MIN_V1_OVER_PEAK);
  if (pool.length < MIN_POOL) {
    pool = complete.filter((v) => v[0] / Math.max(...v) >= 0.05);
  }
  if (pool.length < MIN_POOL) pool = complete;

  const hours = c.forecast.map(({ hour }) => ({
    hour,
    q: pool.length >= MIN_POOL ? hourQuantiles(pool, p1, hour) : baselineQuantiles(p1, hour),
  }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
