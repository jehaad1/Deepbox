/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import type { LegendEntry } from "../types";

/**
 * Normalize an optional legend label: surrounding whitespace is removed, and `undefined`,
 * non-strings and blank labels return null (no legend entry).
 * @internal
 */
export function normalizeLegendLabel(label: string | undefined): string | null {
  if (typeof label !== "string") return null;
  const trimmed = label.trim();
  return trimmed.length > 0 ? trimmed : null;
}

/**
 * Builds a legend entry from a normalized label and its symbol options, or null when the label
 * is null or empty so the drawable stays out of the legend.
 * @internal
 */
export function buildLegendEntry(
  label: string | null,
  entry: Omit<LegendEntry, "label">
): LegendEntry | null {
  if (!label) return null;
  return { label, ...entry };
}
