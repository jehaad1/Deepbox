/**
 * Corrections for multiple comparisons.
 *
 * Each function takes a list of p-values and returns the adjusted p-values and the decision
 * for every hypothesis at a given level.
 *
 * @module stats/multiple
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox documentation}
 */

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
 * @throws {InvalidParameterError} If pvalues is empty or contains a value outside [0, 1], or
 *   alpha is not in (0, 1)
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
 * @throws {InvalidParameterError} If pvalues is empty or contains a value outside [0, 1], or
 *   alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = holm([0.01, 0.04, 0.03, 0.005], 0.05);
 * result.corrected;  // [0.03, 0.06, 0.06, 0.02]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Multiple Comparisons}
 */
export function holm(pvalues: readonly number[], alpha = 0.05): MultipleComparisonResult {
  validateInputs(pvalues, alpha, "holm");
  const m = pvalues.length;
  const indices = sortedOrder(pvalues);

  const corrected = new Array<number>(m);
  let cMax = 0;
  for (let i = 0; i < m; i++) {
    const idx = indices[i] as number;
    const adjusted = Math.min((pvalues[idx] ?? 0) * (m - i), 1);
    cMax = Math.max(cMax, adjusted);
    corrected[idx] = cMax;
  }

  const rejected = corrected.map((p) => p <= alpha);
  return { pvalues, corrected, rejected };
}

/**
 * Hochberg step-up correction for multiple comparisons.
 *
 * Controls the FWER under independence (or positive dependence of the tests) and is more
 * powerful than Holm's method. The adjusted p-value of the i-th smallest p-value is
 * `min over j >= i of (m - j + 1) * p_(j)`.
 *
 * @param pvalues - Array of p-values from individual tests
 * @param alpha - Significance level (default: 0.05)
 * @returns Corrected p-values and rejection decisions
 * @throws {InvalidParameterError} If pvalues is empty or contains a value outside [0, 1], or
 *   alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = hochberg([0.01, 0.04, 0.03, 0.005], 0.05);
 * result.corrected;  // [0.03, 0.04, 0.04, 0.02]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Multiple Comparisons}
 */
export function hochberg(pvalues: readonly number[], alpha = 0.05): MultipleComparisonResult {
  validateInputs(pvalues, alpha, "hochberg");
  const m = pvalues.length;
  const indices = sortedOrder(pvalues);

  const corrected = new Array<number>(m);
  let cMin = 1;
  for (let i = m - 1; i >= 0; i--) {
    const idx = indices[i] as number;
    cMin = Math.min(cMin, (pvalues[idx] ?? 0) * (m - i));
    corrected[idx] = cMin;
  }

  const rejected = corrected.map((p) => p <= alpha);
  return { pvalues, corrected, rejected };
}

/**
 * Benjamini-Hochberg (BH) procedure for controlling the false discovery rate (FDR).
 *
 * A step-up procedure that controls the expected proportion of false positives
 * among rejected hypotheses. More powerful than FWER-controlling methods when
 * many tests are performed. Valid for independent or positively dependent tests.
 *
 * @param pvalues - Array of p-values from individual tests
 * @param alpha - Significance level / FDR level (default: 0.05)
 * @returns Corrected p-values (q-values) and rejection decisions
 * @throws {InvalidParameterError} If pvalues is empty or contains a value outside [0, 1], or
 *   alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = benjaminiHochberg([0.01, 0.04, 0.03, 0.005], 0.05);
 * result.corrected;  // [0.02, 0.04, 0.04, 0.02]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Multiple Comparisons}
 */
export function benjaminiHochberg(
  pvalues: readonly number[],
  alpha = 0.05
): MultipleComparisonResult {
  validateInputs(pvalues, alpha, "benjaminiHochberg");
  return fdrStepUp(pvalues, alpha, 1);
}

/**
 * Benjamini-Yekutieli (BY) procedure for controlling the false discovery rate (FDR).
 *
 * Like {@link benjaminiHochberg}, but valid under arbitrary dependence between the tests. It
 * divides the BH threshold by `1 + 1/2 + ... + 1/m`, so it is more conservative.
 *
 * @param pvalues - Array of p-values from individual tests
 * @param alpha - FDR level (default: 0.05)
 * @returns Corrected p-values (q-values) and rejection decisions
 * @throws {InvalidParameterError} If pvalues is empty or contains a value outside [0, 1], or
 *   alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = benjaminiYekutieli([0.01, 0.04, 0.03, 0.005], 0.05);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox Multiple Comparisons}
 */
export function benjaminiYekutieli(
  pvalues: readonly number[],
  alpha = 0.05
): MultipleComparisonResult {
  validateInputs(pvalues, alpha, "benjaminiYekutieli");
  let harmonic = 0;
  for (let i = pvalues.length; i >= 1; i--) harmonic += 1 / i;
  return fdrStepUp(pvalues, alpha, harmonic);
}

/** Step-up FDR adjustment with the penalty `penalty` on the BH multiplier `m / rank`. */
function fdrStepUp(
  pvalues: readonly number[],
  alpha: number,
  penalty: number
): MultipleComparisonResult {
  const m = pvalues.length;
  const indices = sortedOrder(pvalues);

  const corrected = new Array<number>(m);
  let cMin = 1;
  for (let i = m - 1; i >= 0; i--) {
    const idx = indices[i] as number;
    const adjusted = Math.min(((pvalues[idx] ?? 0) * m * penalty) / (i + 1), 1);
    cMin = Math.min(cMin, adjusted);
    corrected[idx] = cMin;
  }

  const rejected = corrected.map((p) => p <= alpha);
  return { pvalues, corrected, rejected };
}

/**
 * Sidak correction for multiple comparisons.
 *
 * Assumes independence of tests and provides a less conservative correction
 * than Bonferroni. Uses: p_corrected = 1 - (1 - p)^m
 *
 * @param pvalues - Array of p-values from individual tests
 * @param alpha - Significance level (default: 0.05)
 * @returns Corrected p-values and rejection decisions
 * @throws {InvalidParameterError} If pvalues is empty or contains a value outside [0, 1], or
 *   alpha is not in (0, 1)
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
  // 1 - (1 - p)^m written with log1p and expm1, which keeps full precision for small p.
  const corrected = pvalues.map((p) => Math.min(-Math.expm1(m * Math.log1p(-p)), 1));
  const rejected = corrected.map((p) => p <= alpha);
  return { pvalues, corrected, rejected };
}

/** Indices that sort the p-values in ascending order (stable). */
function sortedOrder(pvalues: readonly number[]): Int32Array {
  const m = pvalues.length;
  const indices = new Int32Array(m);
  for (let i = 0; i < m; i++) indices[i] = i;
  indices.sort((a, b) => (pvalues[a] as number) - (pvalues[b] as number) || a - b);
  return indices;
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
