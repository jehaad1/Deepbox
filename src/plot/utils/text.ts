/**
 * Rough text width estimate for SVG layout: 0.6 em per UTF-16 code unit.
 *
 * This is deliberately a cheap upper-ish bound used to size legend boxes. It returns 0 for an
 * empty string or a non-finite or non-positive font size.
 * @internal
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */
export function estimateTextWidth(text: string, fontSize: number): number {
  if (text.length === 0) return 0;
  if (!Number.isFinite(fontSize) || fontSize <= 0) return 0;
  return text.length * fontSize * 0.6;
}
