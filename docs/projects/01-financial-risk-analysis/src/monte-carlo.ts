/**
 * Monte Carlo Simulation Module
 *
 * Implements Monte Carlo methods for portfolio simulation and stress testing.
 * Demonstrates deepbox/random and deepbox/ndarray usage.
 */

import { cholesky } from "deepbox/linalg";
import { type Tensor, tensor } from "deepbox/ndarray";
import { mean, std } from "deepbox/stats";

/**
 * Monte Carlo simulation result
 */
export interface MonteCarloResult {
  meanReturn: number;
  medianReturn: number;
  volatility: number;
  var95: number;
  var99: number;
  percentile5: number;
  percentile25: number;
  percentile75: number;
  percentile95: number;
  scenarios: number[];
}

/**
 * Seeded random number generator for reproducibility
 */
class SeededRandom {
  private seed: number;

  constructor(seed = 42) {
    this.seed = seed;
  }

  /**
   * Generate a random number between 0 and 1
   */
  random(): number {
    this.seed = (this.seed * 1103515245 + 12345) & 0x7fffffff;
    return this.seed / 0x7fffffff;
  }

  /**
   * Generate a standard normal random variable using Box-Muller
   */
  randn(): number {
    const u1 = this.random();
    const u2 = this.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  /**
   * Generate n standard normal random variables
   */
  randnArray(n: number): number[] {
    return Array.from({ length: n }, () => this.randn());
  }

  /**
   * Generate correlated random variables using Cholesky decomposition
   */
  correlatedRandn(covMatrix: Tensor): number[] {
    const n = covMatrix.shape[0];
    const uncorrelated = this.randnArray(n);

    // Get Cholesky decomposition: Σ = L * L'
    const L = cholesky(covMatrix);

    // Correlated = L * uncorrelated
    const correlated: number[] = [];
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j <= i; j++) {
        sum += Number(L.data[i * n + j]) * uncorrelated[j];
      }
      correlated.push(sum);
    }

    return correlated;
  }
}

/**
 * Run Monte Carlo simulation for portfolio returns
 *
 * @param expectedReturns - Expected return for each asset
 * @param covMatrix - Covariance matrix
 * @param weights - Portfolio weights
 * @param numSimulations - Number of simulation runs
 * @param horizon - Time horizon in periods
 * @param seed - Random seed for reproducibility
 * @returns Simulation results
 */
export function simulatePortfolioReturns(
  expectedReturns: number[],
  covMatrix: Tensor,
  weights: number[],
  numSimulations = 10000,
  horizon = 12,
  seed = 42
): MonteCarloResult {
  const rng = new SeededRandom(seed);
  const n = weights.length;
  const scenarios: number[] = [];

  for (let sim = 0; sim < numSimulations; sim++) {
    let cumulativeReturn = 1.0;

    for (let t = 0; t < horizon; t++) {
      // Generate correlated asset returns
      const randomShocks = rng.correlatedRandn(covMatrix);

      // Calculate period return for each asset
      let portfolioReturn = 0;
      for (let i = 0; i < n; i++) {
        const assetReturn = expectedReturns[i] + randomShocks[i];
        portfolioReturn += weights[i] * assetReturn;
      }

      cumulativeReturn *= 1 + portfolioReturn;
    }

    scenarios.push(cumulativeReturn - 1); // Convert to return
  }

  // Sort scenarios for percentile calculations
  const sortedScenarios = [...scenarios].sort((a, b) => a - b);

  // Calculate statistics
  const scenariosTensor = tensor(scenarios);
  const meanReturn = Number(mean(scenariosTensor).data[0]);
  const volatility = Number(std(scenariosTensor).data[0]);

  // Median
  const medianIdx = Math.floor(numSimulations / 2);
  const medianReturn = sortedScenarios[medianIdx];

  // Percentiles
  const percentile5 = sortedScenarios[Math.floor(0.05 * numSimulations)];
  const percentile25 = sortedScenarios[Math.floor(0.25 * numSimulations)];
  const percentile75 = sortedScenarios[Math.floor(0.75 * numSimulations)];
  const percentile95 = sortedScenarios[Math.floor(0.95 * numSimulations)];

  // VaR (negative of percentile for losses)
  const var95 = -sortedScenarios[Math.floor(0.05 * numSimulations)];
  const var99 = -sortedScenarios[Math.floor(0.01 * numSimulations)];

  return {
    meanReturn,
    medianReturn,
    volatility,
    var95,
    var99,
    percentile5,
    percentile25,
    percentile75,
    percentile95,
    scenarios,
  };
}

/**
 * Geometric Brownian Motion simulation for asset prices
 *
 * dS = μS dt + σS dW
 *
 * @param initialPrice - Starting price
 * @param drift - Annual drift (expected return)
 * @param volatility - Annual volatility
 * @param timeHorizon - Time horizon in years
 * @param numSteps - Number of time steps
 * @param numPaths - Number of simulation paths
 * @param seed - Random seed
 * @returns Array of price paths
 */
export function simulateGBM(
  initialPrice: number,
  drift: number,
  volatility: number,
  timeHorizon: number,
  numSteps: number,
  numPaths: number,
  seed = 42
): number[][] {
  const rng = new SeededRandom(seed);
  const dt = timeHorizon / numSteps;
  const sqrtDt = Math.sqrt(dt);

  const paths: number[][] = [];

  for (let path = 0; path < numPaths; path++) {
    const prices: number[] = [initialPrice];
    let currentPrice = initialPrice;

    for (let step = 0; step < numSteps; step++) {
      const dW = rng.randn() * sqrtDt;
      const dS = drift * currentPrice * dt + volatility * currentPrice * dW;
      currentPrice += dS;
      prices.push(Math.max(0, currentPrice)); // Price can't go negative
    }

    paths.push(prices);
  }

  return paths;
}

/**
 * Stress testing scenarios
 */
export interface StressScenario {
  name: string;
  shocks: number[]; // Percentage shocks to each asset
}

/**
 * Run stress tests on portfolio
 *
 * @param weights - Portfolio weights
 * @param expectedReturns - Base expected returns
 * @param scenarios - Stress scenarios to test
 * @returns Results for each scenario
 */
export function runStressTests(
  weights: number[],
  _expectedReturns: number[],
  scenarios: StressScenario[]
): { scenario: string; portfolioImpact: number }[] {
  const results: { scenario: string; portfolioImpact: number }[] = [];

  for (const scenario of scenarios) {
    let portfolioImpact = 0;
    for (let i = 0; i < weights.length; i++) {
      const shock = scenario.shocks[i] || 0;
      portfolioImpact += weights[i] * shock;
    }
    results.push({
      scenario: scenario.name,
      portfolioImpact,
    });
  }

  return results;
}

/**
 * Generate historical stress scenarios based on market events
 */
export function getHistoricalStressScenarios(): StressScenario[] {
  return [
    {
      name: "2008 Financial Crisis",
      shocks: [-0.45, -0.25, -0.55, -0.35, -0.3, 0.05, -0.4, -0.5],
    },
    {
      name: "2020 COVID Crash",
      shocks: [-0.35, 0.1, -0.4, -0.6, -0.25, 0.03, -0.3, -0.35],
    },
    {
      name: "2022 Rate Hikes",
      shocks: [-0.3, -0.15, -0.25, 0.3, -0.2, -0.15, -0.25, -0.2],
    },
    {
      name: "Tech Bubble Burst",
      shocks: [-0.7, -0.2, -0.3, -0.1, -0.15, 0.08, -0.2, -0.4],
    },
    {
      name: "Mild Recession",
      shocks: [-0.15, -0.1, -0.2, -0.15, -0.12, 0.02, -0.15, -0.18],
    },
    {
      name: "Inflation Spike",
      shocks: [-0.2, 0.05, -0.1, 0.25, -0.15, -0.1, 0.1, -0.15],
    },
  ];
}

/**
 * Bootstrap resampling for confidence intervals
 *
 * @param returns - Historical returns
 * @param numBootstrap - Number of bootstrap samples
 * @param statistic - Function to compute statistic
 * @param seed - Random seed
 * @returns Bootstrap distribution of statistic
 */
export function bootstrapReturns(
  returns: number[],
  numBootstrap: number,
  statistic: (data: number[]) => number,
  seed = 42
): number[] {
  const rng = new SeededRandom(seed);
  const n = returns.length;
  const bootstrapStats: number[] = [];

  for (let b = 0; b < numBootstrap; b++) {
    // Resample with replacement
    const resample: number[] = [];
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(rng.random() * n);
      resample.push(returns[idx]);
    }
    bootstrapStats.push(statistic(resample));
  }

  return bootstrapStats;
}

/**
 * Calculate confidence interval from bootstrap samples
 */
export function bootstrapConfidenceInterval(
  bootstrapStats: number[],
  confidenceLevel = 0.95
): { lower: number; upper: number; mean: number } {
  const sorted = [...bootstrapStats].sort((a, b) => a - b);
  const alpha = (1 - confidenceLevel) / 2;
  const n = sorted.length;

  return {
    lower: sorted[Math.floor(alpha * n)],
    upper: sorted[Math.floor((1 - alpha) * n)],
    mean: sorted.reduce((a, b) => a + b, 0) / n,
  };
}
