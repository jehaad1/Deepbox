import type { LegendEntry } from "../types";

/**
 * Normalize an optional legend label, returning null if not usable.
 * @internal
 */
export function normalizeLegendLabel(label: string | undefined): string | null {
  if (typeof label !== "string") return null;
  const trimmed = label.trim();
  return trimmed.length > 0 ? trimmed : null;
}

/**
 * @internal
 */
export function buildLegendEntry(
  label: string | null,
  entry: Omit<LegendEntry, "label">
): LegendEntry | null {
  if (!label) return null;
  return { label, ...entry };
}
