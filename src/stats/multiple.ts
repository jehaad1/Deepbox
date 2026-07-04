import { InvalidParameterError } from "../core";

/**
 * Result of a multiple comparison correction.
 */
export interface MultipleComparisonResult {
  /** Original (uncorrected) p-values */
  pvalues: readonly number[];
  /** Corrected p-values */
  corrected: readonly number[];
  /** Boolean array indicating which hypotheses are rejected at given alpha */
  rejected: readonly boolean[];
}

/**
 * Bonferroni correction for multiple comparisons.
 *
 * The simplest and most conservative method. Multiplies each p-value by
 * the number of tests performed. Controls the family-wise error rate (FWER).
 *
 * @param pvalues - Array of p-values from individual tests
 * @param alpha - Significance level (default: 0.05)
 * @returns Corrected p-values and rejection decisions
 * @throws {InvalidParameterError} If pvalues is empty or alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = bonferroni([0.01, 0.04, 0.03, 0.005], 0.05);
 * result.corrected;  // [0.04, 0.16, 0.12, 0.02]
 * result.rejected;   // [true, false, false, true]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Multiple Comparisons}
 */
export function bonferroni(pvalues: readonly number[], alpha = 0.05): MultipleComparisonResult {
  validateInputs(pvalues, alpha, "bonferroni");
  const m = pvalues.length;
  const corrected = pvalues.map((p) => Math.min(p * m, 1));
  const rejected = corrected.map((p) => p <= alpha);
  return { pvalues, corrected, rejected };
}

/**
 * Holm-Bonferroni step-down correction for multiple comparisons.
 *
 * A sequentially rejective method that is uniformly more powerful than
 * Bonferroni while still controlling the FWER. Sorts p-values and applies
 * decreasing multipliers.
 *
 * @param pvalues - Array of p-values from individual tests
 * @param alpha - Significance level (default: 0.05)
 * @returns Corrected p-values and rejection decisions
 * @throws {InvalidParameterError} If pvalues is empty or alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = holm([0.01, 0.04, 0.03, 0.005], 0.05);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Multiple Comparisons}
 */
export function holm(pvalues: readonly number[], alpha = 0.05): MultipleComparisonResult {
  validateInputs(pvalues, alpha, "holm");
  const m = pvalues.length;
  const indices = Array.from({ length: m }, (_, i) => i);
  indices.sort((a, b) => (pvalues[a] ?? 0) - (pvalues[b] ?? 0));

  const corrected = new Array<number>(m);
  let cMax = 0;
  for (let i = 0; i < m; i++) {
    const idx = indices[i];
    if (idx === undefined) continue;
    const adjusted = Math.min((pvalues[idx] ?? 0) * (m - i), 1);
    cMax = Math.max(cMax, adjusted);
    corrected[idx] = cMax;
  }

  const rejected = corrected.map((p) => p <= alpha);
  return { pvalues, corrected, rejected };
}

/**
 * Benjamini-Hochberg (BH) procedure for controlling the false discovery rate (FDR).
 *
 * A step-up procedure that controls the expected proportion of false positives
 * among rejected hypotheses. More powerful than FWER-controlling methods when
 * many tests are performed.
 *
 * @param pvalues - Array of p-values from individual tests
 * @param alpha - Significance level / FDR level (default: 0.05)
 * @returns Corrected p-values and rejection decisions
 * @throws {InvalidParameterError} If pvalues is empty or alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = benjaminiHochberg([0.01, 0.04, 0.03, 0.005], 0.05);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Multiple Comparisons}
 */
export function benjaminiHochberg(
  pvalues: readonly number[],
  alpha = 0.05
): MultipleComparisonResult {
  validateInputs(pvalues, alpha, "benjaminiHochberg");
  const m = pvalues.length;
  const indices = Array.from({ length: m }, (_, i) => i);
  indices.sort((a, b) => (pvalues[a] ?? 0) - (pvalues[b] ?? 0));

  const corrected = new Array<number>(m);
  let cMin = 1;
  for (let i = m - 1; i >= 0; i--) {
    const idx = indices[i];
    if (idx === undefined) continue;
    const rank = i + 1;
    const adjusted = Math.min(((pvalues[idx] ?? 0) * m) / rank, 1);
    cMin = Math.min(cMin, adjusted);
    corrected[idx] = cMin;
  }

  const rejected = corrected.map((p) => p <= alpha);
  return { pvalues, corrected, rejected };
}

/**
 * Šidák correction for multiple comparisons.
 *
 * Assumes independence of tests and provides a less conservative correction
 * than Bonferroni. Uses: p_corrected = 1 - (1 - p)^m
 *
 * @param pvalues - Array of p-values from individual tests
 * @param alpha - Significance level (default: 0.05)
 * @returns Corrected p-values and rejection decisions
 * @throws {InvalidParameterError} If pvalues is empty or alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = sidak([0.01, 0.04, 0.03, 0.005], 0.05);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Multiple Comparisons}
 */
export function sidak(pvalues: readonly number[], alpha = 0.05): MultipleComparisonResult {
  validateInputs(pvalues, alpha, "sidak");
  const m = pvalues.length;
  const corrected = pvalues.map((p) => Math.min(1 - (1 - p) ** m, 1));
  const rejected = corrected.map((p) => p <= alpha);
  return { pvalues, corrected, rejected };
}

/**
 * Validates input parameters for multiple comparison functions.
 * @internal
 */
function validateInputs(pvalues: readonly number[], alpha: number, fnName: string): void {
  if (pvalues.length === 0) {
    throw new InvalidParameterError(
      `${fnName}() requires at least one p-value`,
      "pvalues",
      pvalues.length
    );
  }
  if (!Number.isFinite(alpha) || alpha <= 0 || alpha >= 1) {
    throw new InvalidParameterError(`${fnName}() requires alpha in (0, 1)`, "alpha", alpha);
  }
  for (let i = 0; i < pvalues.length; i++) {
    const p = pvalues[i];
    if (p === undefined || !Number.isFinite(p) || p < 0 || p > 1) {
      throw new InvalidParameterError(`${fnName}() requires p-values in [0, 1]`, "pvalues", p);
    }
  }
}
