/**
 * Portfolio Management Module
 *
 * Handles portfolio construction, returns calculation, and basic analytics.
 * Demonstrates deepbox/ndarray and deepbox/dataframe usage.
 */

import { InvalidParameterError, isNumericTypedArray, isTypedArray } from "deepbox/core";
import { DataFrame } from "deepbox/dataframe";
import { reshape, type Tensor, tensor } from "deepbox/ndarray";
import { cov, mean, std } from "deepbox/stats";

const expectNumericTypedArray = (
  value: unknown
): Float32Array | Float64Array | Int32Array | Uint8Array => {
  if (!isTypedArray(value) || !isNumericTypedArray(value)) {
    throw new Error("Expected numeric typed array");
  }
  return value;
};

/**
 * Asset data structure representing a single financial asset
 */
export interface Asset {
  symbol: string;
  name: string;
  sector: string;
  returns: number[];
}

/**
 * Portfolio weights allocation
 */
export interface PortfolioWeights {
  [symbol: string]: number;
}

/**
 * Portfolio performance metrics
 */
export interface PortfolioMetrics {
  expectedReturn: number;
  volatility: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
}

/**
 * Portfolio class for managing a collection of assets
 */
export class Portfolio {
  private assets: Asset[];
  private weights: number[];
  private returnsTensor: Tensor;
  private riskFreeRate: number;

  constructor(assets: Asset[], weights: number[], riskFreeRate = 0.02) {
    if (assets.length !== weights.length) {
      throw new InvalidParameterError(
        "Assets and weights must have the same length",
        "weights",
        weights.length
      );
    }

    const weightSum = weights.reduce((a, b) => a + b, 0);
    if (Math.abs(weightSum - 1.0) > 1e-6) {
      throw new InvalidParameterError(
        `Weights must sum to 1.0, got ${weightSum}`,
        "weights",
        weights
      );
    }

    this.assets = assets;
    this.weights = weights;
    this.riskFreeRate = riskFreeRate;

    // Build returns matrix: [n_periods, n_assets]
    const numPeriods = assets[0].returns.length;
    const returnsData: number[][] = [];
    for (let t = 0; t < numPeriods; t++) {
      const periodReturns: number[] = [];
      for (const asset of assets) {
        periodReturns.push(asset.returns[t]);
      }
      returnsData.push(periodReturns);
    }
    this.returnsTensor = tensor(returnsData);
  }

  /**
   * Calculate portfolio returns for each period
   */
  getPortfolioReturns(): Tensor {
    const portfolioReturns: number[] = [];

    for (let t = 0; t < this.returnsTensor.shape[0]; t++) {
      let periodReturn = 0;
      for (let i = 0; i < this.weights.length; i++) {
        const idx = t * this.returnsTensor.shape[1] + i;
        periodReturn += Number(this.returnsTensor.data[idx]) * this.weights[i];
      }
      portfolioReturns.push(periodReturn);
    }

    return tensor(portfolioReturns);
  }

  /**
   * Calculate expected annual return
   */
  getExpectedReturn(): number {
    const portfolioReturns = this.getPortfolioReturns();
    const meanReturn = mean(portfolioReturns);
    // Annualize assuming monthly returns (multiply by 12)
    return Number(meanReturn.data[0]) * 12;
  }

  /**
   * Calculate portfolio volatility (annualized standard deviation)
   */
  getVolatility(): number {
    const portfolioReturns = this.getPortfolioReturns();
    const stdDev = std(portfolioReturns);
    // Annualize assuming monthly returns (multiply by sqrt(12))
    return Number(stdDev.data[0]) * Math.sqrt(12);
  }

  /**
   * Calculate Sharpe Ratio
   * (Expected Return - Risk Free Rate) / Volatility
   */
  getSharpeRatio(): number {
    const expectedReturn = this.getExpectedReturn();
    const volatility = this.getVolatility();
    if (volatility === 0) return 0;
    return (expectedReturn - this.riskFreeRate) / volatility;
  }

  /**
   * Calculate Sortino Ratio (uses downside deviation)
   */
  getSortinoRatio(): number {
    const portfolioReturns = this.getPortfolioReturns();
    const expectedReturn = this.getExpectedReturn();

    // Calculate downside deviation (only negative returns)
    const returnsArray = Array.from(expectNumericTypedArray(portfolioReturns.data));
    const monthlyRiskFree = this.riskFreeRate / 12;
    const negativeReturns = returnsArray.filter((r) => r < monthlyRiskFree);

    if (negativeReturns.length === 0) return Number.POSITIVE_INFINITY;

    const downsideVariance =
      negativeReturns.reduce((sum, r) => sum + (r - monthlyRiskFree) ** 2, 0) /
      negativeReturns.length;
    const downsideDeviation = Math.sqrt(downsideVariance) * Math.sqrt(12);

    if (downsideDeviation === 0) return Number.POSITIVE_INFINITY;
    return (expectedReturn - this.riskFreeRate) / downsideDeviation;
  }

  /**
   * Calculate maximum drawdown
   */
  getMaxDrawdown(): number {
    const portfolioReturns = this.getPortfolioReturns();
    const returnsArray = Array.from(expectNumericTypedArray(portfolioReturns.data));

    // Calculate cumulative returns
    let cumulativeValue = 1.0;
    const values: number[] = [1.0];
    for (const r of returnsArray) {
      cumulativeValue *= 1 + r;
      values.push(cumulativeValue);
    }

    // Calculate drawdowns
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
   * Get all portfolio metrics
   */
  getMetrics(): PortfolioMetrics {
    return {
      expectedReturn: this.getExpectedReturn(),
      volatility: this.getVolatility(),
      sharpeRatio: this.getSharpeRatio(),
      sortinoRatio: this.getSortinoRatio(),
      maxDrawdown: this.getMaxDrawdown(),
    };
  }

  /**
   * Get covariance matrix of asset returns
   */
  getCovarianceMatrix(): Tensor {
    return cov(this.returnsTensor);
  }

  /**
   * Get correlation matrix of asset returns
   */
  getCorrelationMatrix(): Tensor {
    const covMatrix = this.getCovarianceMatrix();
    const n = covMatrix.shape[0];
    const corrData: number[] = [];

    // Extract standard deviations from diagonal
    const stdDevs: number[] = [];
    for (let i = 0; i < n; i++) {
      const variance = Number(covMatrix.data[i * n + i]);
      stdDevs.push(Math.sqrt(variance));
    }

    // Calculate correlation matrix
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const covIJ = Number(covMatrix.data[i * n + j]);
        const corr = stdDevs[i] * stdDevs[j] === 0 ? 0 : covIJ / (stdDevs[i] * stdDevs[j]);
        corrData.push(corr);
      }
    }

    return reshape(tensor(corrData), [n, n]);
  }

  /**
   * Get asset symbols
   */
  getSymbols(): string[] {
    return this.assets.map((a) => a.symbol);
  }

  /**
   * Get weights
   */
  getWeights(): number[] {
    return [...this.weights];
  }

  /**
   * Get returns tensor
   */
  getReturnsTensor(): Tensor {
    return this.returnsTensor;
  }

  /**
   * Create a DataFrame summary of the portfolio
   */
  toDataFrame(): DataFrame {
    return new DataFrame({
      symbol: this.assets.map((a) => a.symbol),
      name: this.assets.map((a) => a.name),
      sector: this.assets.map((a) => a.sector),
      weight: this.weights.map((w) => w * 100),
      avgReturn: this.assets.map((a) => {
        const meanRet = a.returns.reduce((s, r) => s + r, 0) / a.returns.length;
        return meanRet * 100;
      }),
      volatility: this.assets.map((a) => {
        const meanRet = a.returns.reduce((s, r) => s + r, 0) / a.returns.length;
        const variance = a.returns.reduce((s, r) => s + (r - meanRet) ** 2, 0) / a.returns.length;
        return Math.sqrt(variance) * 100;
      }),
    });
  }
}

/**
 * Generate synthetic asset data for testing
 */
export function generateSyntheticAssets(numPeriods = 60): Asset[] {
  // Simulate realistic asset characteristics
  const assetConfigs = [
    {
      symbol: "TECH",
      name: "Technology ETF",
      sector: "Technology",
      meanReturn: 0.015,
      volatility: 0.06,
    },
    {
      symbol: "HLTH",
      name: "Healthcare ETF",
      sector: "Healthcare",
      meanReturn: 0.01,
      volatility: 0.04,
    },
    {
      symbol: "FINC",
      name: "Financial ETF",
      sector: "Financials",
      meanReturn: 0.008,
      volatility: 0.05,
    },
    {
      symbol: "ENGY",
      name: "Energy ETF",
      sector: "Energy",
      meanReturn: 0.005,
      volatility: 0.08,
    },
    {
      symbol: "CONS",
      name: "Consumer ETF",
      sector: "Consumer",
      meanReturn: 0.009,
      volatility: 0.035,
    },
    {
      symbol: "BOND",
      name: "Bond ETF",
      sector: "Fixed Income",
      meanReturn: 0.003,
      volatility: 0.015,
    },
    {
      symbol: "REIT",
      name: "Real Estate ETF",
      sector: "Real Estate",
      meanReturn: 0.007,
      volatility: 0.045,
    },
    {
      symbol: "INTL",
      name: "International ETF",
      sector: "International",
      meanReturn: 0.006,
      volatility: 0.055,
    },
  ];

  // Simple seeded random number generator for reproducibility
  let seed = 42;
  const seededRandom = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };

  // Box-Muller transform for normal distribution
  const randomNormal = (mean: number, std: number) => {
    const u1 = seededRandom();
    const u2 = seededRandom();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
  };

  return assetConfigs.map((config) => ({
    symbol: config.symbol,
    name: config.name,
    sector: config.sector,
    returns: Array.from({ length: numPeriods }, () =>
      randomNormal(config.meanReturn, config.volatility)
    ),
  }));
}
