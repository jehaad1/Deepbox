/**
 * Tick generation for linear and logarithmic axes.
 * @internal
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */
export type Tick = {
  readonly value: number;
  readonly label: string;
};

/** Upper bound on the number of ticks a caller may request (guards against runaway loops). */
const MAX_TICK_REQUEST = 1000;

function niceNumber(range: number, round: boolean): number {
  if (!Number.isFinite(range) || range <= 0) return 1;
  const exponent = Math.floor(Math.log10(range));
  const fraction = range / 10 ** exponent;
  let niceFraction: number;
  if (round) {
    if (fraction < 1.5) niceFraction = 1;
    else if (fraction < 3) niceFraction = 2;
    else if (fraction < 4.5) niceFraction = 2.5;
    else if (fraction < 7) niceFraction = 5;
    else niceFraction = 10;
  } else {
    if (fraction <= 1) niceFraction = 1;
    else if (fraction <= 2) niceFraction = 2;
    else if (fraction <= 2.5) niceFraction = 2.5;
    else if (fraction <= 5) niceFraction = 5;
    else niceFraction = 10;
  }
  return niceFraction * 10 ** exponent;
}

/**
 * Smallest number of decimals (at most 10) that shows `step` exactly, e.g. 0 for 5, 1 for 0.5
 * and 2 for 0.25.
 */
function decimalsForStep(step: number): number {
  const abs = Math.abs(step);
  for (let d = 0; d < 10; d++) {
    const scaled = abs * 10 ** d;
    if (Math.abs(scaled - Math.round(scaled)) <= 1e-9 * Math.max(1, scaled)) return d;
  }
  return 10;
}

function formatTick(value: number, step: number): string {
  if (!Number.isFinite(value)) return "";
  const abs = Math.abs(value);
  const absStep = Math.abs(step);
  if (abs === 0 || abs < absStep * 1e-9) return "0";
  if (abs < 1e-4 || abs >= 1e6) {
    // Show enough mantissa digits to tell neighbouring ticks apart (e.g. 1e9 + 2 and 1e9 + 4),
    // never fewer than two.
    const needed = Math.ceil(Math.log10(abs / absStep));
    return value.toExponential(Math.min(15, Math.max(2, needed)));
  }

  const decimals = decimalsForStep(step);
  let text = value.toFixed(decimals);
  if (decimals > 0) {
    text = text.replace(/\.?0+$/, "");
  }
  return text;
}

/** The next smaller step of the 1, 2, 2.5, 5 times a power of ten sequence. */
function finerStep(step: number): number {
  const exponent = Math.floor(Math.log10(step) + 1e-9);
  const base = 10 ** exponent;
  const fraction = Math.round((step / base) * 100) / 100;
  if (fraction <= 1) return 5 * 10 ** (exponent - 1);
  if (fraction <= 2) return base;
  if (fraction <= 2.5) return 2 * base;
  return 2.5 * base;
}

/**
 * Generate "nice" ticks for an axis range.
 *
 * Tick values are multiples of a step drawn from 1, 2, 2.5, 5 times a power of ten, computed as
 * `k * step` (not by repeated addition) so they carry no accumulated rounding error. Only ticks
 * inside `[min, max]` are returned. Reversed bounds are accepted; a zero-width range is widened
 * around the value. Returns an empty array for non-finite bounds or a non-positive `maxTicks`.
 *
 * The step is chosen so that about `maxTicks` ticks fit, but never so coarse that fewer than two
 * ticks fall inside the range (for example the integer positions 1, 2, 3 of a bar chart with a
 * small `maxTicks`): the step is then refined, as matplotlib's `MaxNLocator` does with
 * `min_n_ticks=2`, so `maxTicks` is a target and not a strict upper bound for narrow ranges.
 * @internal
 */
export function generateTicks(min: number, max: number, maxTicks = 5): readonly Tick[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (!(maxTicks > 0) || !Number.isFinite(maxTicks)) return [];
  const m = Math.min(min, max);
  const M = Math.max(min, max);
  if (m === M) {
    const span = Math.max(1, Math.abs(m) * 0.05);
    return generateTicks(m - span, M + span, maxTicks);
  }
  const requested = Math.min(MAX_TICK_REQUEST, maxTicks);
  const range = niceNumber(M - m, false);
  let step = niceNumber(range / Math.max(1, requested - 1), true);
  if (!Number.isFinite(step) || step <= 0) return [];
  const tol = 1e-9;
  let first = Math.ceil(m / step - tol);
  let last = Math.floor(M / step + tol);
  // Refine a step that is so coarse that fewer than two ticks fall inside the range.
  for (let i = 0; i < 4 && last - first < 1; i++) {
    const finer = finerStep(step);
    const f = Math.ceil(m / finer - tol);
    const l = Math.floor(M / finer + tol);
    if (!Number.isSafeInteger(f) || !Number.isSafeInteger(l)) break;
    step = finer;
    first = f;
    last = l;
  }
  if (!Number.isSafeInteger(first) || !Number.isSafeInteger(last) || last - first > 10_000) {
    // The step is below the floating-point resolution at this magnitude (or absurdly small
    // against the range), so a tick grid cannot be represented. Label the end points only.
    return [m, M].map((v) => ({ value: v, label: formatTick(v, M - m) }));
  }
  const ticks: Tick[] = [];
  for (let k = first; k <= last; k++) {
    // toPrecision(15) removes representation noise such as 0.30000000000000004 and turns -0
    // into 0. When the value is so large against the step that 15 digits would merge
    // neighbouring ticks, the product is already as exact as a double allows and is kept.
    const raw = k * step;
    const v = Math.abs(raw) < step * 1e14 ? Number(raw.toPrecision(15)) : raw;
    ticks.push({ value: v, label: formatTick(v, step) });
  }
  return ticks;
}

function logTickLabel(mantissa: number, exponent: number): string {
  const value = Number(`${mantissa}e${exponent}`);
  if (exponent >= -4 && exponent <= 6) return String(value);
  return mantissa === 1 ? `1e${exponent}` : `${mantissa}e${exponent}`;
}

/**
 * Log-scale ticks: one per decade (..., 0.1, 1, 10, 100, ...) with the data value
 * as the label. `min`/`max` are the DATA-space bounds; only positive bounds
 * are meaningful on a log axis, so anything else returns an empty array.
 *
 * - When more than `maxTicks` decades are visible, every n-th decade is kept (counted from
 *   exponent 0) so labels stay readable.
 * - When the range covers less than two whole decades, ticks at 1, 2 and 5 times each power of
 *   ten are used instead so the axis is never left without labels.
 * @internal
 */
export function generateLogTicks(min: number, max: number, maxTicks = 10): readonly Tick[] {
  const lo = Math.min(min, max);
  const hi = Math.max(min, max);
  if (!(lo > 0) || !(hi > 0) || !Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  const startExp = Math.floor(Math.log10(lo));
  const endExp = Math.ceil(Math.log10(hi));
  const inRange = (v: number): boolean => v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9);

  const decades: Tick[] = [];
  for (let e = startExp; e <= endExp; e++) {
    const v = Number(`1e${e}`);
    if (!inRange(v)) continue;
    decades.push({ value: v, label: logTickLabel(1, e) });
  }

  if (decades.length < 2) {
    const sub: Tick[] = [];
    for (let e = startExp; e <= endExp; e++) {
      for (const mantissa of [1, 2, 5]) {
        const v = Number(`${mantissa}e${e}`);
        if (v > 0 && Number.isFinite(v) && inRange(v)) {
          sub.push({ value: v, label: logTickLabel(mantissa, e) });
        }
      }
    }
    return sub.length > decades.length ? sub : decades;
  }

  const limit = Math.max(1, Math.min(MAX_TICK_REQUEST, Math.floor(maxTicks) || 1));
  if (decades.length <= limit) return decades;
  const every = Math.ceil(decades.length / limit);
  return decades.filter((t) => {
    const e = Math.round(Math.log10(t.value));
    return ((e % every) + every) % every === 0;
  });
}
