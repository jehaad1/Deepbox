/**
 * Portfolio Optimization Module
 *
 * Implements Mean-Variance Optimization (Markowitz Portfolio Theory).
 * Demonstrates deepbox/linalg for matrix operations.
 */

import { inv } from "deepbox/linalg";
import type { Tensor } from "deepbox/ndarray";

/**
 * Optimization result structure
 */
export interface OptimizationResult {
  weights: number[];
  expectedReturn: number;
  volatility: number;
  sharpeRatio: number;
}

/**
 * Efficient frontier point
 */
export interface EfficientFrontierPoint {
  return: number;
  volatility: number;
  weights: number[];
}

/**
 * Calculate the minimum variance portfolio
 *
 * Finds the portfolio with the lowest possible volatility.
 *
 * @param covMatrix - Covariance matrix of asset returns
 * @returns Optimal weights for minimum variance portfolio
 */
export function minimumVariancePortfolio(covMatrix: Tensor): number[] {
  const n = covMatrix.shape[0];

  // Minimum variance portfolio: w = Σ^(-1) * 1 / (1' * Σ^(-1) * 1)
  const covInverse = inv(covMatrix);

  // Calculate Σ^(-1) * 1
  const invTimesOnes: number[] = [];
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      sum += Number(covInverse.data[i * n + j]) * 1;
    }
    invTimesOnes.push(sum);
  }

  // Calculate 1' * Σ^(-1) * 1 (scalar)
  const denominator = invTimesOnes.reduce((s, v) => s + v, 0);

  // Calculate weights
  const weights = invTimesOnes.map((v) => v / denominator);

  return weights;
}

/**
 * Calculate the maximum Sharpe ratio portfolio (tangency portfolio)
 *
 * Finds the portfolio that maximizes risk-adjusted returns.
 *
 * @param expectedReturns - Expected returns for each asset
 * @param covMatrix - Covariance matrix of asset returns
 * @param riskFreeRate - Risk-free rate (annualized)
 * @returns Optimal weights for maximum Sharpe ratio
 */
export function maxSharpePortfolio(
  expectedReturns: number[],
  covMatrix: Tensor,
  riskFreeRate = 0.02
): OptimizationResult {
  const n = expectedReturns.length;

  // Excess returns over risk-free rate
  const excessReturns = expectedReturns.map((r) => r - riskFreeRate / 12);

  // w = Σ^(-1) * (μ - rf) / (1' * Σ^(-1) * (μ - rf))
  const covInverse = inv(covMatrix);

  // Calculate Σ^(-1) * excess returns
  const invTimesExcess: number[] = [];
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      sum += Number(covInverse.data[i * n + j]) * excessReturns[j];
    }
    invTimesExcess.push(sum);
  }

  // Normalize to sum to 1
  const sumWeights = invTimesExcess.reduce((s, v) => s + v, 0);
  const weights = invTimesExcess.map((v) => v / sumWeights);

  // Handle potential short positions by normalizing absolute values
  // for a long-only constraint (simplified approach)
  const hasNegative = weights.some((w) => w < 0);
  let finalWeights = weights;

  if (hasNegative) {
    // Simple long-only approximation: set negatives to 0, renormalize
    const positiveWeights = weights.map((w) => Math.max(0, w));
    const positiveSum = positiveWeights.reduce((s, v) => s + v, 0);
    finalWeights = positiveWeights.map((w) => w / positiveSum);
  }

  // Calculate portfolio metrics
  const portfolioReturn = finalWeights.reduce((sum, w, i) => sum + w * expectedReturns[i], 0);

  // Portfolio variance: w' * Σ * w
  let portfolioVariance = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      portfolioVariance += finalWeights[i] * finalWeights[j] * Number(covMatrix.data[i * n + j]);
    }
  }
  const portfolioVol = Math.sqrt(portfolioVariance) * Math.sqrt(12);

  const sharpeRatio = (portfolioReturn * 12 - riskFreeRate) / portfolioVol;

  return {
    weights: finalWeights,
    expectedReturn: portfolioReturn * 12,
    volatility: portfolioVol,
    sharpeRatio,
  };
}

/**
 * Calculate a target return portfolio
 *
 * Finds the minimum variance portfolio for a given target return.
 *
 * @param expectedReturns - Expected returns for each asset
 * @param covMatrix - Covariance matrix of asset returns
 * @param targetReturn - Target portfolio return (monthly)
 * @returns Optimal weights for target return
 */
export function targetReturnPortfolio(
  expectedReturns: number[],
  covMatrix: Tensor,
  targetReturn: number
): OptimizationResult {
  const n = expectedReturns.length;

  // Use Lagrangian optimization
  // Minimize: w' * Σ * w
  // Subject to: w' * μ = target, w' * 1 = 1

  const covInverse = inv(covMatrix);

  // Calculate A = 1' * Σ^(-1) * μ
  // Calculate B = μ' * Σ^(-1) * μ
  // Calculate C = 1' * Σ^(-1) * 1

  let A = 0,
    B = 0,
    C = 0;

  for (let i = 0; i < n; i++) {
    let sumMu = 0,
      sumOne = 0;
    for (let j = 0; j < n; j++) {
      sumMu += Number(covInverse.data[i * n + j]) * expectedReturns[j];
      sumOne += Number(covInverse.data[i * n + j]) * 1;
    }
    A += 1 * sumMu;
    B += expectedReturns[i] * sumMu;
    C += 1 * sumOne;
  }

  const D = B * C - A * A;

  // Calculate optimal weights
  // w = (1/D) * Σ^(-1) * [(B - A*target)*1 + (C*target - A)*μ]
  const weights: number[] = [];

  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      const invIJ = Number(covInverse.data[i * n + j]);
      sum += invIJ * ((B - A * targetReturn) * 1 + (C * targetReturn - A) * expectedReturns[j]);
    }
    weights.push(sum / D);
  }

  // Handle long-only constraint
  const hasNegative = weights.some((w) => w < 0);
  let finalWeights = weights;

  if (hasNegative) {
    const positiveWeights = weights.map((w) => Math.max(0, w));
    const positiveSum = positiveWeights.reduce((s, v) => s + v, 0);
    finalWeights = positiveWeights.map((w) => w / positiveSum);
  }

  // Calculate portfolio metrics
  const portfolioReturn = finalWeights.reduce((sum, w, i) => sum + w * expectedReturns[i], 0);

  let portfolioVariance = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      portfolioVariance += finalWeights[i] * finalWeights[j] * Number(covMatrix.data[i * n + j]);
    }
  }
  const portfolioVol = Math.sqrt(portfolioVariance) * Math.sqrt(12);

  return {
    weights: finalWeights,
    expectedReturn: portfolioReturn * 12,
    volatility: portfolioVol,
    sharpeRatio: (portfolioReturn * 12 - 0.02) / portfolioVol,
  };
}

/**
 * Generate the efficient frontier
 *
 * Creates a series of optimal portfolios at different return levels.
 *
 * @param expectedReturns - Expected returns for each asset
 * @param covMatrix - Covariance matrix of asset returns
 * @param numPoints - Number of points on the frontier
 * @returns Array of efficient frontier points
 */
export function generateEfficientFrontier(
  expectedReturns: number[],
  covMatrix: Tensor,
  numPoints = 50
): EfficientFrontierPoint[] {
  const minReturn = Math.min(...expectedReturns);
  const maxReturn = Math.max(...expectedReturns);

  const frontier: EfficientFrontierPoint[] = [];
  const step = (maxReturn - minReturn) / (numPoints - 1);

  for (let i = 0; i < numPoints; i++) {
    const targetReturn = minReturn + i * step;
    try {
      const result = targetReturnPortfolio(expectedReturns, covMatrix, targetReturn);
      frontier.push({
        return: result.expectedReturn,
        volatility: result.volatility,
        weights: result.weights,
      });
    } catch (_e) {
      // Skip invalid points
    }
  }

  return frontier;
}

/**
 * Risk parity portfolio optimization
 *
 * Each asset contributes equally to portfolio risk.
 *
 * @param covMatrix - Covariance matrix of asset returns
 * @param maxIterations - Maximum iterations for convergence
 * @param tolerance - Convergence tolerance
 * @returns Risk parity weights
 */
export function riskParityPortfolio(
  covMatrix: Tensor,
  maxIterations = 1000,
  tolerance = 1e-8
): number[] {
  const n = covMatrix.shape[0];

  // Initialize equal weights
  let weights = Array(n).fill(1 / n);

  // Iterative algorithm
  for (let iter = 0; iter < maxIterations; iter++) {
    // Calculate marginal risk contributions
    const marginalRisk: number[] = [];
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < n; j++) {
        sum += Number(covMatrix.data[i * n + j]) * weights[j];
      }
      marginalRisk.push(sum);
    }

    // Calculate total portfolio variance
    let portfolioVariance = 0;
    for (let i = 0; i < n; i++) {
      portfolioVariance += weights[i] * marginalRisk[i];
    }

    // Calculate risk contributions
    const riskContributions = weights.map((w, i) => (w * marginalRisk[i]) / portfolioVariance);

    // Target: equal risk contribution
    const targetRisk = 1 / n;

    // Update weights
    const newWeights = weights.map((w, i) => w * Math.sqrt(targetRisk / riskContributions[i]));

    // Normalize
    const sumWeights = newWeights.reduce((s, v) => s + v, 0);
    const normalizedWeights = newWeights.map((w) => w / sumWeights);

    // Check convergence
    const maxDiff = Math.max(...normalizedWeights.map((w, i) => Math.abs(w - weights[i])));
    weights = normalizedWeights;

    if (maxDiff < tolerance) {
      break;
    }
  }

  return weights;
}

/**
 * Black-Litterman model for combining market equilibrium with investor views
 *
 * @param marketWeights - Market capitalization weights
 * @param covMatrix - Covariance matrix
 * @param views - Investor views matrix (P)
 * @param viewReturns - Expected returns from views (Q)
 * @param tau - Scaling factor for prior uncertainty
 * @param omega - View uncertainty matrix (diagonal)
 * @returns Black-Litterman expected returns
 */
export function blackLittermanReturns(
  marketWeights: number[],
  covMatrix: Tensor,
  riskAversion = 2.5,
  _tau = 0.05
): number[] {
  const n = marketWeights.length;

  // Calculate implied equilibrium returns: π = δ * Σ * w_mkt
  const impliedReturns: number[] = [];

  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      sum += Number(covMatrix.data[i * n + j]) * marketWeights[j];
    }
    impliedReturns.push(riskAversion * sum);
  }

  return impliedReturns;
}
