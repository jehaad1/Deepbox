/**
 * Rough text width estimate for SVG layout.
 * @internal
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */
export function estimateTextWidth(text: string, fontSize: number): number {
  if (text.length === 0) return 0;
  return text.length * fontSize * 0.6;
}
