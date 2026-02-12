/**
 * Time Series Stock Price Forecasting
 *
 * Demonstrates time series analysis and forecasting using Deepbox.
 *
 * Deepbox Modules Used:
 * - deepbox/ndarray: Tensor operations
 * - deepbox/stats: Statistical analysis
 * - deepbox/ml: Regression models
 * - deepbox/metrics: Forecasting metrics
 * - deepbox/dataframe: Data manipulation
 * - deepbox/plot: Visualization
 */

import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { isNumericTypedArray, isTypedArray } from "deepbox/core";
import { DataFrame } from "deepbox/dataframe";
import { mae, mse, r2Score, rmse } from "deepbox/metrics";
import { LinearRegression, Ridge } from "deepbox/ml";
import { tensor } from "deepbox/ndarray";
import { Figure } from "deepbox/plot";
import { StandardScaler } from "deepbox/preprocess";
import { mean, pearsonr, std } from "deepbox/stats";

// ============================================================================
// Configuration
// ============================================================================

const OUTPUT_DIR = "docs/projects/04-stock-price-forecasting/output";
const NUM_DAYS = 500;
const LOOKBACK = 20;

const expectNumericTypedArray = (
  value: unknown
): Float32Array | Float64Array | Int32Array | Uint8Array => {
  if (!isTypedArray(value) || !isNumericTypedArray(value)) {
    throw new Error("Expected numeric typed array");
  }
  return value;
};

// ============================================================================
// Data Generation
// ============================================================================

/**
 * Generate synthetic stock price data with realistic patterns
 */
function generateStockData(
  numDays: number,
  seed = 42
): {
  dates: string[];
  prices: number[];
  returns: number[];
  volume: number[];
} {
  let randomSeed = seed;
  const seededRandom = () => {
    randomSeed = (randomSeed * 1103515245 + 12345) & 0x7fffffff;
    return randomSeed / 0x7fffffff;
  };

  const randomNormal = (mean: number, std: number) => {
    const u1 = seededRandom();
    const u2 = seededRandom();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
  };

  const dates: string[] = [];
  const prices: number[] = [];
  const returns: number[] = [];
  const volume: number[] = [];

  let price = 100; // Starting price
  const startDate = new Date("2023-01-01");

  for (let i = 0; i < numDays; i++) {
    // Generate date
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    dates.push(date.toISOString().split("T")[0]);

    // Generate return with trend, volatility, and mean reversion
    const trend = 0.0002; // Slight upward trend
    const volatility = 0.02;
    const meanReversion = (-0.01 * (price - 100)) / 100; // Pull back to 100

    const dailyReturn = trend + meanReversion + randomNormal(0, volatility);
    returns.push(dailyReturn);

    // Update price
    price = price * (1 + dailyReturn);
    prices.push(price);

    // Generate volume with some correlation to volatility
    const baseVolume = 1000000;
    const volumeMultiplier = 1 + Math.abs(dailyReturn) * 10;
    volume.push(Math.round(baseVolume * volumeMultiplier * (0.8 + seededRandom() * 0.4)));
  }

  return { dates, prices, returns, volume };
}

/**
 * Calculate technical indicators
 */
function calculateIndicators(
  prices: number[],
  returns: number[]
): {
  sma20: number[];
  sma50: number[];
  volatility20: number[];
  rsi14: number[];
  momentum10: number[];
} {
  const n = prices.length;

  // Simple Moving Averages
  const sma20: number[] = [];
  const sma50: number[] = [];
  for (let i = 0; i < n; i++) {
    if (i >= 19) {
      const window = prices.slice(i - 19, i + 1);
      sma20.push(window.reduce((a, b) => a + b, 0) / 20);
    } else {
      sma20.push(NaN);
    }
    if (i >= 49) {
      const window = prices.slice(i - 49, i + 1);
      sma50.push(window.reduce((a, b) => a + b, 0) / 50);
    } else {
      sma50.push(NaN);
    }
  }

  // Rolling Volatility (20-day)
  const volatility20: number[] = [];
  for (let i = 0; i < n; i++) {
    if (i >= 19) {
      const window = returns.slice(i - 19, i + 1);
      const mean = window.reduce((a, b) => a + b, 0) / 20;
      const variance = window.reduce((sum, r) => sum + (r - mean) ** 2, 0) / 20;
      volatility20.push(Math.sqrt(variance) * Math.sqrt(252)); // Annualized
    } else {
      volatility20.push(NaN);
    }
  }

  // RSI (14-day)
  const rsi14: number[] = [];
  for (let i = 0; i < n; i++) {
    if (i >= 14) {
      const gains: number[] = [];
      const losses: number[] = [];
      for (let j = i - 13; j <= i; j++) {
        if (returns[j] > 0) {
          gains.push(returns[j]);
          losses.push(0);
        } else {
          gains.push(0);
          losses.push(-returns[j]);
        }
      }
      const avgGain = gains.reduce((a, b) => a + b, 0) / 14;
      const avgLoss = losses.reduce((a, b) => a + b, 0) / 14;
      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      rsi14.push(100 - 100 / (1 + rs));
    } else {
      rsi14.push(NaN);
    }
  }

  // Momentum (10-day price change)
  const momentum10: number[] = [];
  for (let i = 0; i < n; i++) {
    if (i >= 10) {
      momentum10.push((prices[i] - prices[i - 10]) / prices[i - 10]);
    } else {
      momentum10.push(NaN);
    }
  }

  return { sma20, sma50, volatility20, rsi14, momentum10 };
}

/**
 * Create features for forecasting
 */
function createFeatures(
  prices: number[],
  returns: number[],
  indicators: ReturnType<typeof calculateIndicators>,
  lookback: number
): { X: number[][]; y: number[]; validIndices: number[] } {
  const X: number[][] = [];
  const y: number[] = [];
  const validIndices: number[] = [];

  const startIdx = Math.max(lookback, 50); // Ensure all indicators are available

  for (let i = startIdx; i < prices.length - 1; i++) {
    // Check if all indicators are valid
    if (
      Number.isNaN(indicators.sma20[i]) ||
      Number.isNaN(indicators.sma50[i]) ||
      Number.isNaN(indicators.volatility20[i]) ||
      Number.isNaN(indicators.rsi14[i]) ||
      Number.isNaN(indicators.momentum10[i])
    ) {
      continue;
    }

    const features: number[] = [];

    // Lagged returns
    for (let j = 0; j < lookback; j++) {
      features.push(returns[i - j]);
    }

    // Technical indicators
    features.push((prices[i] - indicators.sma20[i]) / indicators.sma20[i]); // Price vs SMA20
    features.push((prices[i] - indicators.sma50[i]) / indicators.sma50[i]); // Price vs SMA50
    features.push(indicators.volatility20[i]);
    features.push(indicators.rsi14[i] / 100); // Normalize RSI
    features.push(indicators.momentum10[i]);

    X.push(features);
    y.push(returns[i + 1]); // Next day return
    validIndices.push(i);
  }

  return { X, y, validIndices };
}

// ============================================================================
// Main Execution
// ============================================================================

console.log("═".repeat(70));
console.log("  TIME SERIES STOCK PRICE FORECASTING");
console.log("  Built with Deepbox - TypeScript Data Science & ML Library");
console.log("═".repeat(70));

// Create output directory
if (!existsSync(OUTPUT_DIR)) {
  mkdirSync(OUTPUT_DIR, { recursive: true });
}

// ============================================================================
// Step 1: Generate Data
// ============================================================================

console.log("\n📊 STEP 1: Generating Stock Price Data");
console.log("─".repeat(70));

const { dates, prices, returns, volume: _volume } = generateStockData(NUM_DAYS);

console.log(`\n✓ Generated ${NUM_DAYS} days of synthetic stock data`);
console.log(`  Date Range: ${dates[0]} to ${dates[dates.length - 1]}`);
console.log(`  Starting Price: $${prices[0].toFixed(2)}`);
console.log(`  Ending Price: $${prices[prices.length - 1].toFixed(2)}`);
console.log(`  Total Return: ${((prices[prices.length - 1] / prices[0] - 1) * 100).toFixed(2)}%`);

// Basic statistics
const returnsTensor = tensor(returns);
const meanReturn = Number(mean(returnsTensor).data[0]);
const stdReturn = Number(std(returnsTensor).data[0]);

console.log(`\nReturn Statistics:`);
console.log(`  Mean Daily Return: ${(meanReturn * 100).toFixed(4)}%`);
console.log(`  Daily Volatility:  ${(stdReturn * 100).toFixed(4)}%`);
console.log(`  Annualized Return: ${(meanReturn * 252 * 100).toFixed(2)}%`);
console.log(`  Annualized Vol:    ${(stdReturn * Math.sqrt(252) * 100).toFixed(2)}%`);

// ============================================================================
// Step 2: Calculate Technical Indicators
// ============================================================================

console.log("\n📈 STEP 2: Calculating Technical Indicators");
console.log("─".repeat(70));

const indicators = calculateIndicators(prices, returns);

// Show sample of indicators
const sampleIdx = prices.length - 1;
console.log(`\nLatest Indicators (${dates[sampleIdx]}):`);
console.log(`  Price:       $${prices[sampleIdx].toFixed(2)}`);
console.log(`  SMA(20):     $${indicators.sma20[sampleIdx].toFixed(2)}`);
console.log(`  SMA(50):     $${indicators.sma50[sampleIdx].toFixed(2)}`);
console.log(`  Volatility:  ${(indicators.volatility20[sampleIdx] * 100).toFixed(2)}%`);
console.log(`  RSI(14):     ${indicators.rsi14[sampleIdx].toFixed(2)}`);
console.log(`  Momentum:    ${(indicators.momentum10[sampleIdx] * 100).toFixed(2)}%`);

// ============================================================================
// Step 3: Feature Engineering
// ============================================================================

console.log("\n🔧 STEP 3: Feature Engineering");
console.log("─".repeat(70));

const { X, y, validIndices: _validIndices } = createFeatures(prices, returns, indicators, LOOKBACK);

console.log(`\n✓ Created feature matrix`);
console.log(`  Samples: ${X.length}`);
console.log(`  Features per sample: ${X[0].length}`);
console.log(`  Feature breakdown:`);
console.log(`    - ${LOOKBACK} lagged returns`);
console.log(`    - 5 technical indicators`);

// ============================================================================
// Step 4: Train/Test Split
// ============================================================================

console.log("\n📦 STEP 4: Train/Test Split");
console.log("─".repeat(70));

const splitIdx = Math.floor(X.length * 0.8);
const XTrain = X.slice(0, splitIdx);
const XTest = X.slice(splitIdx);
const yTrain = y.slice(0, splitIdx);
const yTest = y.slice(splitIdx);

console.log(`\n✓ Time-based split (no shuffle to preserve temporal order)`);
console.log(`  Training: ${XTrain.length} samples`);
console.log(`  Testing:  ${XTest.length} samples`);

// Scale features
const scaler = new StandardScaler();
scaler.fit(tensor(XTrain));
const XTrainScaled = scaler.transform(tensor(XTrain));
const XTestScaled = scaler.transform(tensor(XTest));

console.log(`✓ Applied StandardScaler`);

// ============================================================================
// Step 5: Model Training
// ============================================================================

console.log("\n🤖 STEP 5: Model Training");
console.log("─".repeat(70));

// Linear Regression
console.log("\nTraining Linear Regression...");
const lr = new LinearRegression();
lr.fit(XTrainScaled, tensor(yTrain));
const yPredLR = lr.predict(XTestScaled);

// Ridge Regression
console.log("Training Ridge Regression (alpha=0.1)...");
const ridge = new Ridge({ alpha: 0.1 });
ridge.fit(XTrainScaled, tensor(yTrain));
const yPredRidge = ridge.predict(XTestScaled);

// Baseline: Predict mean
const meanPred = yTrain.reduce((a, b) => a + b, 0) / yTrain.length;
const yPredBaseline = tensor(Array(yTest.length).fill(meanPred));

console.log("✓ Models trained");

// ============================================================================
// Step 6: Model Evaluation
// ============================================================================

console.log("\n📊 STEP 6: Model Evaluation");
console.log("─".repeat(70));

const yTestTensor = tensor(yTest);

// Calculate metrics
const results = [
  {
    name: "Mean Baseline",
    mse: mse(yTestTensor, yPredBaseline),
    rmse: rmse(yTestTensor, yPredBaseline),
    mae: mae(yTestTensor, yPredBaseline),
    r2: r2Score(yTestTensor, yPredBaseline),
    pred: yPredBaseline,
  },
  {
    name: "Linear Regression",
    mse: mse(yTestTensor, yPredLR),
    rmse: rmse(yTestTensor, yPredLR),
    mae: mae(yTestTensor, yPredLR),
    r2: r2Score(yTestTensor, yPredLR),
    pred: yPredLR,
  },
  {
    name: "Ridge Regression",
    mse: mse(yTestTensor, yPredRidge),
    rmse: rmse(yTestTensor, yPredRidge),
    mae: mae(yTestTensor, yPredRidge),
    r2: r2Score(yTestTensor, yPredRidge),
    pred: yPredRidge,
  },
];

console.log("\nModel Comparison:\n");
const metricsDF = new DataFrame({
  Model: results.map((r) => r.name),
  "MSE (×10⁻⁵)": results.map((r) => (Number(r.mse) * 100000).toFixed(4)),
  "RMSE (%)": results.map((r) => (Number(r.rmse) * 100).toFixed(4)),
  "MAE (%)": results.map((r) => (Number(r.mae) * 100).toFixed(4)),
  "R² Score": results.map((r) => Number(r.r2).toFixed(4)),
});
console.log(metricsDF.toString());

// Directional accuracy
console.log("\nDirectional Accuracy (predicting up/down):");
for (const result of results) {
  const predData = expectNumericTypedArray(result.pred.data);
  let correct = 0;
  for (let i = 0; i < yTest.length; i++) {
    const actualDirection = yTest[i] > 0 ? 1 : -1;
    const predDirection = predData[i] > 0 ? 1 : -1;
    if (actualDirection === predDirection) correct++;
  }
  const dirAcc = correct / yTest.length;
  console.log(`  ${result.name.padEnd(20)}: ${(dirAcc * 100).toFixed(2)}%`);
}

// ============================================================================
// Step 7: Autocorrelation Analysis
// ============================================================================

console.log("\n🔍 STEP 7: Autocorrelation Analysis");
console.log("─".repeat(70));

// Calculate autocorrelation of returns
console.log("\nReturn Autocorrelation:");
for (const lag of [1, 5, 10, 20]) {
  const returns1 = returns.slice(0, returns.length - lag);
  const returns2 = returns.slice(lag);
  const [corr] = pearsonr(tensor(returns1), tensor(returns2));
  console.log(`  Lag ${String(lag).padStart(2)}: ${corr.toFixed(4)}`);
}

// ============================================================================
// Step 8: Visualizations
// ============================================================================

console.log("\n📊 STEP 8: Generating Visualizations");
console.log("─".repeat(70));

// Price chart
try {
  const fig = new Figure({ width: 1000, height: 400 });
  const ax = fig.addAxes();

  const xValues = Array.from({ length: prices.length }, (_, i) => i);
  ax.plot(tensor(xValues), tensor(prices), { color: "#2196F3", linewidth: 1 });
  ax.setTitle("Stock Price Over Time");
  ax.setXLabel("Day");
  ax.setYLabel("Price ($)");

  const svg = fig.renderSVG();
  writeFileSync(`${OUTPUT_DIR}/price-chart.svg`, svg.svg);
  console.log(`  ✓ Saved: ${OUTPUT_DIR}/price-chart.svg`);
} catch (e) {
  console.log(`  ⚠ Could not generate price chart: ${e}`);
}

// Returns distribution
try {
  const fig = new Figure({ width: 800, height: 400 });
  const ax = fig.addAxes();

  // Create histogram
  const numBins = 30;
  const minRet = Math.min(...returns);
  const maxRet = Math.max(...returns);
  const binWidth = (maxRet - minRet) / numBins;

  const bins: number[] = [];
  const counts: number[] = [];
  for (let i = 0; i < numBins; i++) {
    const binCenter = minRet + (i + 0.5) * binWidth;
    const count = returns.filter(
      (r) => r >= minRet + i * binWidth && r < minRet + (i + 1) * binWidth
    ).length;
    bins.push(binCenter * 100);
    counts.push(count);
  }

  ax.bar(tensor(bins), tensor(counts), { color: "#4CAF50" });
  ax.setTitle("Daily Returns Distribution");
  ax.setXLabel("Return (%)");
  ax.setYLabel("Frequency");

  const svg = fig.renderSVG();
  writeFileSync(`${OUTPUT_DIR}/returns-distribution.svg`, svg.svg);
  console.log(`  ✓ Saved: ${OUTPUT_DIR}/returns-distribution.svg`);
} catch (e) {
  console.log(`  ⚠ Could not generate returns distribution: ${e}`);
}

// ============================================================================
// Step 9: Summary
// ============================================================================

console.log(`\n${"═".repeat(70)}`);
console.log("  FORECASTING COMPLETE - SUMMARY");
console.log("═".repeat(70));

const bestModel = results.reduce((best, r) => (Number(r.r2) > Number(best.r2) ? r : best));

console.log("\n📌 Key Findings:\n");
console.log("  1. Data Overview:");
console.log(`     • ${NUM_DAYS} days of price data`);
console.log(`     • Annualized return: ${(meanReturn * 252 * 100).toFixed(2)}%`);
console.log(`     • Annualized volatility: ${(stdReturn * Math.sqrt(252) * 100).toFixed(2)}%`);

console.log("\n  2. Best Model:");
console.log(`     • ${bestModel.name}`);
console.log(`     • R² Score: ${Number(bestModel.r2).toFixed(4)}`);
console.log(`     • RMSE: ${(Number(bestModel.rmse) * 100).toFixed(4)}%`);

console.log("\n  3. Observations:");
console.log("     • Stock returns show low autocorrelation (efficient market)");
console.log("     • Technical indicators provide marginal improvement");
console.log("     • Directional prediction is challenging (~50% baseline)");

console.log("\n📁 Output Files:");
console.log(`   • ${OUTPUT_DIR}/price-chart.svg`);
console.log(`   • ${OUTPUT_DIR}/returns-distribution.svg`);

console.log(`\n${"═".repeat(70)}`);
console.log("  ✅ Stock Price Forecasting Complete!");
console.log("═".repeat(70));
