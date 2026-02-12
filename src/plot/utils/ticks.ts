/**
 * @internal
 */
export type Tick = {
  readonly value: number;
  readonly label: string;
};

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

function formatTick(value: number, step: number): string {
  if (!Number.isFinite(value)) return "";
  const abs = Math.abs(value);
  if (abs < 1e-12) return "0";
  if ((abs > 0 && abs < 1e-4) || abs >= 1e6) {
    return value.toExponential(2);
  }

  const absStep = Math.abs(step);
  let decimals = 0;
  if (absStep > 0 && absStep < 1) {
    decimals = Math.min(6, Math.ceil(-Math.log10(absStep)));
  }
  let text = value.toFixed(decimals);
  if (decimals > 0) {
    text = text.replace(/\.?0+$/, "");
  }
  return text;
}

/**
 * Generate "nice" ticks for an axis range.
 * @internal
 */
export function generateTicks(min: number, max: number, maxTicks = 5): readonly Tick[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (maxTicks <= 0 || !Number.isFinite(maxTicks)) return [];
  const m = Math.min(min, max);
  const M = Math.max(min, max);
  if (m === M) {
    const span = Math.max(1, Math.abs(m) * 0.05);
    return generateTicks(m - span, M + span, maxTicks);
  }
  const range = niceNumber(M - m, false);
  const step = niceNumber(range / Math.max(1, maxTicks - 1), true);
  if (!Number.isFinite(step) || step <= 0) return [];
  const niceMin = Math.floor(m / step) * step;
  const niceMax = Math.ceil(M / step) * step;
  const ticks: Tick[] = [];
  const epsilon = step * 1e-9;
  for (let v = niceMin; v <= niceMax + epsilon; v += step) {
    if (v + epsilon < m || v - epsilon > M) continue;
    ticks.push({ value: v, label: formatTick(v, step) });
  }
  return ticks;
}
