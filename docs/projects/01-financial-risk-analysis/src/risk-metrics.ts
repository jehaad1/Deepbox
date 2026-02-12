/**
 * Risk Metrics Module
 *
 * Calculates Value at Risk (VaR), Conditional VaR (CVaR), and other risk metrics.
 * Demonstrates deepbox/stats and deepbox/ndarray usage.
 */

import { InvalidParameterError, isNumericTypedArray, isTypedArray } from "deepbox/core";
import { type Tensor, tensor } from "deepbox/ndarray";
import { mean, std } from "deepbox/stats";

const expectNumericTypedArray = (
  value: unknown
): Float32Array | Float64Array | Int32Array | Uint8Array => {
  if (!isTypedArray(value) || !isNumericTypedArray(value)) {
    throw new Error("Expected numeric typed array");
  }
  return value;
};

/**
 * VaR calculation methods
 */
export type VaRMethod = "historical" | "parametric" | "cornish-fisher";

/**
 * Risk metrics result structure
 */
export interface RiskMetricsResult {
  var95: number;
  var99: number;
  cvar95: number;
  cvar99: number;
  volatility: number;
  downsideDeviation: number;
  maxDrawdown: number;
  calmarRatio: number;
}

/**
 * Calculate Value at Risk (VaR) using historical simulation
 *
 * VaR represents the maximum expected loss at a given confidence level
 * over a specified time horizon.
 *
 * @param returns - Array or Tensor of historical returns
 * @param confidenceLevel - Confidence level (e.g., 0.95 for 95%)
 * @returns VaR value (positive number representing potential loss)
 */
export function calculateVaR(
  returns: number[] | Tensor,
  confidenceLevel = 0.95,
  method: VaRMethod = "historical"
): number {
  const returnsArray = Array.isArray(returns)
    ? returns
    : Array.from(expectNumericTypedArray(returns.data));

  if (method === "historical") {
    // Historical VaR: Use empirical percentile
    const sortedReturns = [...returnsArray].sort((a, b) => a - b);
    const index = Math.floor((1 - confidenceLevel) * sortedReturns.length);
    return -sortedReturns[index];
  }

  if (method === "parametric") {
    // Parametric VaR: Assume normal distribution
    const returnsTensor = tensor(returnsArray);
    const meanVal = Number(mean(returnsTensor).data[0]);
    const stdVal = Number(std(returnsTensor).data[0]);

    // Z-scores for common confidence levels
    const zScores: { [key: number]: number } = {
      0.9: 1.282,
      0.95: 1.645,
      0.99: 2.326,
    };
    const zScore = zScores[confidenceLevel] || 1.645;

    return -(meanVal - zScore * stdVal);
  }

  if (method === "cornish-fisher") {
    // Cornish-Fisher VaR: Adjusts for skewness and kurtosis
    const returnsTensor = tensor(returnsArray);
    const meanVal = Number(mean(returnsTensor).data[0]);
    const stdVal = Number(std(returnsTensor).data[0]);

    // Calculate skewness
    const n = returnsArray.length;
    const skewness =
      returnsArray.reduce((sum, r) => sum + ((r - meanVal) / stdVal) ** 3, 0) *
      (n / ((n - 1) * (n - 2)));

    // Calculate excess kurtosis
    const kurtosis =
      returnsArray.reduce((sum, r) => sum + ((r - meanVal) / stdVal) ** 4, 0) *
        ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) -
      (3 * (n - 1) * (n - 1)) / ((n - 2) * (n - 3));

    // Base z-score
    const z = confidenceLevel === 0.99 ? -2.326 : -1.645;

    // Cornish-Fisher expansion
    const zCF =
      z +
      ((z * z - 1) * skewness) / 6 +
      ((z * z * z - 3 * z) * kurtosis) / 24 -
      ((2 * z * z * z - 5 * z) * skewness * skewness) / 36;

    return -(meanVal + zCF * stdVal);
  }

  throw new InvalidParameterError(`Unknown VaR method: ${method}`, "method", method);
}

/**
 * Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
 *
 * CVaR represents the expected loss given that the loss exceeds VaR.
 * It captures tail risk better than VaR.
 *
 * @param returns - Array or Tensor of historical returns
 * @param confidenceLevel - Confidence level (e.g., 0.95 for 95%)
 * @returns CVaR value (positive number representing expected tail loss)
 */
export function calculateCVaR(returns: number[] | Tensor, confidenceLevel = 0.95): number {
  const returnsArray = Array.isArray(returns)
    ? returns
    : Array.from(expectNumericTypedArray(returns.data));

  // Sort returns ascending
  const sortedReturns = [...returnsArray].sort((a, b) => a - b);

  // Find the VaR cutoff index
  const cutoffIndex = Math.floor((1 - confidenceLevel) * sortedReturns.length);

  // CVaR is the average of returns below VaR
  if (cutoffIndex === 0) return -sortedReturns[0];

  const tailReturns = sortedReturns.slice(0, cutoffIndex);
  const avgTailLoss = tailReturns.reduce((sum, r) => sum + r, 0) / tailReturns.length;

  return -avgTailLoss;
}

/**
 * Calculate downside deviation (semi-deviation)
 *
 * Only considers returns below a target (usually 0 or risk-free rate)
 *
 * @param returns - Array or Tensor of historical returns
 * @param target - Target return threshold (default: 0)
 * @returns Downside deviation
 */
export function calculateDownsideDeviation(returns: number[] | Tensor, target = 0): number {
  const returnsArray = Array.isArray(returns)
    ? returns
    : Array.from(expectNumericTypedArray(returns.data));

  const downsideReturns = returnsArray.filter((r) => r < target);
  if (downsideReturns.length === 0) return 0;

  const sumSquaredDeviations = downsideReturns.reduce((sum, r) => sum + (r - target) ** 2, 0);

  return Math.sqrt(sumSquaredDeviations / downsideReturns.length);
}

/**
 * Calculate maximum drawdown from a return series
 *
 * Maximum drawdown is the largest peak-to-trough decline in portfolio value.
 *
 * @param returns - Array or Tensor of historical returns
 * @returns Maximum drawdown as a positive percentage
 */
export function calculateMaxDrawdown(returns: number[] | Tensor): number {
  const returnsArray = Array.isArray(returns)
    ? returns
    : Array.from(expectNumericTypedArray(returns.data));

  // Calculate cumulative returns
  let cumulativeValue = 1.0;
  const values: number[] = [1.0];
  for (const r of returnsArray) {
    cumulativeValue *= 1 + r;
    values.push(cumulativeValue);
  }

  // Find maximum drawdown
  let maxDrawdown = 0;
  let peak = values[0];

  for (const value of values) {
    if (value > peak) peak = value;
    const drawdown = (peak - value) / peak;
    if (drawdown > maxDrawdown) maxDrawdown = drawdown;
  }

  return maxDrawdown;
}

/**
 * Calculate Calmar Ratio
 *
 * Calmar Ratio = Annualized Return / Maximum Drawdown
 *
 * @param returns - Array or Tensor of historical returns
 * @param periodsPerYear - Number of periods per year (e.g., 12 for monthly, 252 for daily)
 * @returns Calmar ratio
 */
export function calculateCalmarRatio(returns: number[] | Tensor, periodsPerYear = 12): number {
  const returnsArray = Array.isArray(returns)
    ? returns
    : Array.from(expectNumericTypedArray(returns.data));

  const returnsTensor = tensor(returnsArray);
  const meanReturn = Number(mean(returnsTensor).data[0]);
  const annualizedReturn = meanReturn * periodsPerYear;

  const maxDrawdown = calculateMaxDrawdown(returns);
  if (maxDrawdown === 0) return Number.POSITIVE_INFINITY;

  return annualizedReturn / maxDrawdown;
}

/**
 * Calculate comprehensive risk metrics
 *
 * @param returns - Array or Tensor of historical returns
 * @param periodsPerYear - Number of periods per year
 * @returns Complete risk metrics object
 */
export function calculateRiskMetrics(
  returns: number[] | Tensor,
  periodsPerYear = 12
): RiskMetricsResult {
  const returnsArray = Array.isArray(returns)
    ? returns
    : Array.from(expectNumericTypedArray(returns.data));

  const returnsTensor = tensor(returnsArray);

  return {
    var95: calculateVaR(returnsArray, 0.95),
    var99: calculateVaR(returnsArray, 0.99),
    cvar95: calculateCVaR(returnsArray, 0.95),
    cvar99: calculateCVaR(returnsArray, 0.99),
    volatility: Number(std(returnsTensor).data[0]) * Math.sqrt(periodsPerYear),
    downsideDeviation: calculateDownsideDeviation(returnsArray) * Math.sqrt(periodsPerYear),
    maxDrawdown: calculateMaxDrawdown(returnsArray),
    calmarRatio: calculateCalmarRatio(returnsArray, periodsPerYear),
  };
}

/**
 * Calculate rolling VaR over a window
 *
 * @param returns - Full return series
 * @param windowSize - Rolling window size
 * @param confidenceLevel - VaR confidence level
 * @returns Array of rolling VaR values
 */
export function calculateRollingVaR(
  returns: number[],
  windowSize: number,
  confidenceLevel = 0.95
): number[] {
  const rollingVaR: number[] = [];

  for (let i = windowSize; i <= returns.length; i++) {
    const window = returns.slice(i - windowSize, i);
    rollingVaR.push(calculateVaR(window, confidenceLevel));
  }

  return rollingVaR;
}

/**
 * Perform VaR backtesting
 *
 * Counts the number of times actual losses exceeded VaR predictions.
 *
 * @param returns - Actual returns
 * @param varValues - Predicted VaR values (aligned with returns)
 * @returns Exceedance count and rate
 */
export function backtestVaR(
  returns: number[],
  varValues: number[]
): { exceedances: number; rate: number; expected: number } {
  if (returns.length !== varValues.length) {
    throw new InvalidParameterError(
      "Returns and VaR arrays must have the same length",
      "varValues",
      varValues.length
    );
  }

  let exceedances = 0;
  for (let i = 0; i < returns.length; i++) {
    if (-returns[i] > varValues[i]) {
      exceedances++;
    }
  }

  return {
    exceedances,
    rate: exceedances / returns.length,
    expected: 0.05, // For 95% VaR
  };
}
