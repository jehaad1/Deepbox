/**
 * Financial Portfolio Risk Analysis System
 *
 * A comprehensive financial risk analysis application demonstrating:
 * - Portfolio construction and management
 * - Risk metrics (VaR, CVaR, Sharpe, Sortino)
 * - Mean-variance optimization
 * - Monte Carlo simulation
 * - Correlation analysis
 * - Stress testing
 *
 * Deepbox Modules Used:
 * - deepbox/ndarray: Tensor operations
 * - deepbox/linalg: Matrix decomposition, inverse
 * - deepbox/stats: Correlation, covariance, statistical measures
 * - deepbox/dataframe: Data manipulation
 * - deepbox/plot: Visualization
 */

import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { isNumericTypedArray, isTypedArray } from "deepbox/core";
import { DataFrame } from "deepbox/dataframe";
import { det, trace } from "deepbox/linalg";
import { tensor } from "deepbox/ndarray";
import { Figure } from "deepbox/plot";
import { pearsonr, shapiro, spearmanr } from "deepbox/stats";
import {
  bootstrapConfidenceInterval,
  bootstrapReturns,
  getHistoricalStressScenarios,
  runStressTests,
  simulatePortfolioReturns,
} from "./src/monte-carlo";
import {
  generateEfficientFrontier,
  maxSharpePortfolio,
  minimumVariancePortfolio,
  riskParityPortfolio,
} from "./src/optimization";
import { generateSyntheticAssets, Portfolio } from "./src/portfolio";
import {
  backtestVaR,
  calculateRiskMetrics,
  calculateRollingVaR,
  calculateVaR,
} from "./src/risk-metrics";

// ============================================================================
// Configuration
// ============================================================================

const OUTPUT_DIR = "docs/projects/01-financial-risk-analysis/output";
const NUM_PERIODS = 60; // 5 years of monthly data
const RISK_FREE_RATE = 0.02; // 2% annual risk-free rate

const expectNumericTypedArray = (
  value: unknown
): Float32Array | Float64Array | Int32Array | Uint8Array => {
  if (!isTypedArray(value) || !isNumericTypedArray(value)) {
    throw new Error("Expected numeric typed array");
  }
  return value;
};

// ============================================================================
// Main Execution
// ============================================================================

console.log("═".repeat(70));
console.log("  FINANCIAL PORTFOLIO RISK ANALYSIS SYSTEM");
console.log("  Built with Deepbox - TypeScript Data Science & ML Library");
console.log("═".repeat(70));

// Create output directory
if (!existsSync(OUTPUT_DIR)) {
  mkdirSync(OUTPUT_DIR, { recursive: true });
}

// ============================================================================
// Step 1: Generate Synthetic Asset Data
// ============================================================================

console.log("\n📊 STEP 1: Generating Asset Data");
console.log("─".repeat(70));

const assets = generateSyntheticAssets(NUM_PERIODS);

console.log(`Generated ${assets.length} assets with ${NUM_PERIODS} periods of data:\n`);
const assetSummary = new DataFrame({
  Symbol: assets.map((a) => a.symbol),
  Name: assets.map((a) => a.name),
  Sector: assets.map((a) => a.sector),
  "Avg Monthly Return (%)": assets.map((a) => {
    const avg = a.returns.reduce((s, r) => s + r, 0) / a.returns.length;
    return (avg * 100).toFixed(3);
  }),
  "Volatility (%)": assets.map((a) => {
    const avg = a.returns.reduce((s, r) => s + r, 0) / a.returns.length;
    const variance = a.returns.reduce((s, r) => s + (r - avg) ** 2, 0) / a.returns.length;
    return (Math.sqrt(variance) * 100).toFixed(3);
  }),
});

console.log(assetSummary.toString());

// ============================================================================
// Step 2: Build Initial Portfolio
// ============================================================================

console.log("\n📈 STEP 2: Building Initial Portfolio");
console.log("─".repeat(70));

// Start with equal weights
const equalWeights = Array(assets.length).fill(1 / assets.length);
const portfolio = new Portfolio(assets, equalWeights, RISK_FREE_RATE);

console.log("\nInitial Equal-Weight Portfolio Allocation:");
const portfolioDF = portfolio.toDataFrame();
console.log(portfolioDF.toString());

const initialMetrics = portfolio.getMetrics();
console.log("\nInitial Portfolio Metrics:");
console.log(`  Expected Annual Return: ${(initialMetrics.expectedReturn * 100).toFixed(2)}%`);
console.log(`  Annual Volatility:      ${(initialMetrics.volatility * 100).toFixed(2)}%`);
console.log(`  Sharpe Ratio:           ${initialMetrics.sharpeRatio.toFixed(3)}`);
console.log(`  Sortino Ratio:          ${initialMetrics.sortinoRatio.toFixed(3)}`);
console.log(`  Maximum Drawdown:       ${(initialMetrics.maxDrawdown * 100).toFixed(2)}%`);

// ============================================================================
// Step 3: Risk Analysis
// ============================================================================

console.log("\n⚠️  STEP 3: Risk Analysis");
console.log("─".repeat(70));

const portfolioReturns = portfolio.getPortfolioReturns();
const returnsArray = Array.from(expectNumericTypedArray(portfolioReturns.data));

const riskMetrics = calculateRiskMetrics(returnsArray);

console.log("\nValue at Risk (VaR):");
console.log(`  95% VaR (Historical): ${(riskMetrics.var95 * 100).toFixed(2)}%`);
console.log(`  99% VaR (Historical): ${(riskMetrics.var99 * 100).toFixed(2)}%`);

console.log("\nConditional Value at Risk (CVaR / Expected Shortfall):");
console.log(`  95% CVaR: ${(riskMetrics.cvar95 * 100).toFixed(2)}%`);
console.log(`  99% CVaR: ${(riskMetrics.cvar99 * 100).toFixed(2)}%`);

console.log("\nOther Risk Metrics:");
console.log(`  Annualized Volatility:    ${(riskMetrics.volatility * 100).toFixed(2)}%`);
console.log(`  Downside Deviation:       ${(riskMetrics.downsideDeviation * 100).toFixed(2)}%`);
console.log(`  Maximum Drawdown:         ${(riskMetrics.maxDrawdown * 100).toFixed(2)}%`);
console.log(`  Calmar Ratio:             ${riskMetrics.calmarRatio.toFixed(3)}`);

// VaR comparison across methods
console.log("\nVaR Method Comparison (95% Confidence):");
console.log(
  `  Historical:     ${(calculateVaR(returnsArray, 0.95, "historical") * 100).toFixed(2)}%`
);
console.log(
  `  Parametric:     ${(calculateVaR(returnsArray, 0.95, "parametric") * 100).toFixed(2)}%`
);
console.log(
  `  Cornish-Fisher: ${(calculateVaR(returnsArray, 0.95, "cornish-fisher") * 100).toFixed(2)}%`
);

// ============================================================================
// Step 4: Correlation Analysis
// ============================================================================

console.log("\n🔗 STEP 4: Correlation Analysis");
console.log("─".repeat(70));

const corrMatrix = portfolio.getCorrelationMatrix();
const covMatrix = portfolio.getCovarianceMatrix();

console.log("\nCorrelation Matrix:");
const symbols = portfolio.getSymbols();
const corrData: { [key: string]: number[] } = {};
for (let i = 0; i < symbols.length; i++) {
  const symbol = symbols[i];
  corrData[symbol] = [];
  for (let j = 0; j < symbols.length; j++) {
    corrData[symbol].push(Number(Number(corrMatrix.data[i * symbols.length + j]).toFixed(3)));
  }
}
const corrDF = new DataFrame(corrData);
console.log(corrDF.toString());

// Covariance matrix properties
console.log("\nCovariance Matrix Properties:");
console.log(`  Determinant: ${det(covMatrix).toExponential(4)}`);
console.log(`  Trace:       ${Number(trace(covMatrix).data[0]).toFixed(6)}`);

// Statistical tests on returns
console.log("\nStatistical Tests (First Asset vs Last Asset):");
const asset1Returns = tensor(assets[0]?.returns);
const asset8Returns = tensor(assets[7]?.returns);

const [pearsonCorr, pearsonP] = pearsonr(asset1Returns, asset8Returns);
const [spearmanCorr, spearmanP] = spearmanr(asset1Returns, asset8Returns);

console.log(`  Pearson Correlation:  ${pearsonCorr.toFixed(4)} (p-value: ${pearsonP.toFixed(4)})`);
console.log(
  `  Spearman Correlation: ${spearmanCorr.toFixed(4)} (p-value: ${spearmanP.toFixed(4)})`
);

// Normality test
const shapiroResult = shapiro(asset1Returns);
console.log(
  `  Shapiro-Wilk Test (TECH): W=${shapiroResult.statistic.toFixed(
    4
  )}, p=${shapiroResult.pvalue.toFixed(4)}`
);

// ============================================================================
// Step 5: Portfolio Optimization
// ============================================================================

console.log("\n🎯 STEP 5: Portfolio Optimization");
console.log("─".repeat(70));

const expectedReturns = assets.map((a) => a.returns.reduce((s, r) => s + r, 0) / a.returns.length);

// Minimum Variance Portfolio
console.log("\n1. Minimum Variance Portfolio:");
const minVarWeights = minimumVariancePortfolio(covMatrix);
const minVarPortfolio = new Portfolio(assets, minVarWeights, RISK_FREE_RATE);
const minVarMetrics = minVarPortfolio.getMetrics();
console.log(`   Weights: [${minVarWeights.map((w) => `${(w * 100).toFixed(1)}%`).join(", ")}]`);
console.log(`   Expected Return: ${(minVarMetrics.expectedReturn * 100).toFixed(2)}%`);
console.log(`   Volatility:      ${(minVarMetrics.volatility * 100).toFixed(2)}%`);
console.log(`   Sharpe Ratio:    ${minVarMetrics.sharpeRatio.toFixed(3)}`);

// Maximum Sharpe Portfolio
console.log("\n2. Maximum Sharpe Ratio Portfolio:");
const maxSharpeResult = maxSharpePortfolio(expectedReturns, covMatrix, RISK_FREE_RATE);
console.log(
  `   Weights: [${maxSharpeResult.weights.map((w) => `${(w * 100).toFixed(1)}%`).join(", ")}]`
);
console.log(`   Expected Return: ${(maxSharpeResult.expectedReturn * 100).toFixed(2)}%`);
console.log(`   Volatility:      ${(maxSharpeResult.volatility * 100).toFixed(2)}%`);
console.log(`   Sharpe Ratio:    ${maxSharpeResult.sharpeRatio.toFixed(3)}`);

// Risk Parity Portfolio
console.log("\n3. Risk Parity Portfolio:");
const riskParityWeights = riskParityPortfolio(covMatrix);
const riskParityPortfolioObj = new Portfolio(assets, riskParityWeights, RISK_FREE_RATE);
const riskParityMetrics = riskParityPortfolioObj.getMetrics();
console.log(`   Weights: [${riskParityWeights.map((w) => `${(w * 100).toFixed(1)}%`).join(", ")}]`);
console.log(`   Expected Return: ${(riskParityMetrics.expectedReturn * 100).toFixed(2)}%`);
console.log(`   Volatility:      ${(riskParityMetrics.volatility * 100).toFixed(2)}%`);
console.log(`   Sharpe Ratio:    ${riskParityMetrics.sharpeRatio.toFixed(3)}`);

// Generate Efficient Frontier
console.log("\n4. Efficient Frontier (sample points):");
const frontier = generateEfficientFrontier(expectedReturns, covMatrix, 10);
const frontierDF = new DataFrame({
  "Return (%)": frontier.map((p) => (p.return * 100).toFixed(2)),
  "Volatility (%)": frontier.map((p) => (p.volatility * 100).toFixed(2)),
});
console.log(frontierDF.toString());

// ============================================================================
// Step 6: Monte Carlo Simulation
// ============================================================================

console.log("\n🎲 STEP 6: Monte Carlo Simulation");
console.log("─".repeat(70));

const optimalWeights = maxSharpeResult.weights;
const mcResult = simulatePortfolioReturns(
  expectedReturns,
  covMatrix,
  optimalWeights,
  10000, // 10,000 simulations
  12, // 12 month horizon
  42 // seed
);

console.log("\nMonte Carlo Simulation Results (10,000 scenarios, 12-month horizon):");
console.log(`  Mean Return:     ${(mcResult.meanReturn * 100).toFixed(2)}%`);
console.log(`  Median Return:   ${(mcResult.medianReturn * 100).toFixed(2)}%`);
console.log(`  Volatility:      ${(mcResult.volatility * 100).toFixed(2)}%`);
console.log(`  95% VaR:         ${(mcResult.var95 * 100).toFixed(2)}%`);
console.log(`  99% VaR:         ${(mcResult.var99 * 100).toFixed(2)}%`);

console.log("\nReturn Distribution Percentiles:");
console.log(`  5th percentile:  ${(mcResult.percentile5 * 100).toFixed(2)}%`);
console.log(`  25th percentile: ${(mcResult.percentile25 * 100).toFixed(2)}%`);
console.log(`  75th percentile: ${(mcResult.percentile75 * 100).toFixed(2)}%`);
console.log(`  95th percentile: ${(mcResult.percentile95 * 100).toFixed(2)}%`);

// Bootstrap confidence interval for Sharpe ratio
console.log("\nBootstrap Confidence Interval for Mean Return:");
const bootstrapMeans = bootstrapReturns(
  returnsArray,
  1000,
  (data) => data.reduce((a, b) => a + b, 0) / data.length,
  42
);
const ci = bootstrapConfidenceInterval(bootstrapMeans, 0.95);
console.log(`  95% CI: [${(ci.lower * 100).toFixed(3)}%, ${(ci.upper * 100).toFixed(3)}%]`);
console.log(`  Bootstrap Mean: ${(ci.mean * 100).toFixed(3)}%`);

// ============================================================================
// Step 7: Stress Testing
// ============================================================================

console.log("\n💥 STEP 7: Stress Testing");
console.log("─".repeat(70));

const stressScenarios = getHistoricalStressScenarios();
const stressResults = runStressTests(optimalWeights, expectedReturns, stressScenarios);

console.log("\nStress Test Results (Optimal Portfolio):\n");
const stressDF = new DataFrame({
  Scenario: stressResults.map((r) => r.scenario),
  "Portfolio Impact (%)": stressResults.map((r) => (r.portfolioImpact * 100).toFixed(2)),
});
console.log(stressDF.toString());

// ============================================================================
// Step 8: VaR Backtesting
// ============================================================================

console.log("\n📋 STEP 8: VaR Backtesting");
console.log("─".repeat(70));

// Calculate rolling VaR
const rollingVaR = calculateRollingVaR(returnsArray, 20, 0.95);

// Backtest against actual returns (offset by window size)
const backtestReturns = returnsArray.slice(20);
const backtestResult = backtestVaR(backtestReturns, rollingVaR.slice(0, backtestReturns.length));

console.log("\nVaR Backtesting Results:");
console.log(`  Number of Periods:    ${backtestReturns.length}`);
console.log(`  VaR Exceedances:      ${backtestResult.exceedances}`);
console.log(`  Exceedance Rate:      ${(backtestResult.rate * 100).toFixed(2)}%`);
console.log(`  Expected Rate (95%):  ${(backtestResult.expected * 100).toFixed(2)}%`);
console.log(
  `  Model ${backtestResult.rate <= 0.07 ? "PASSES" : "FAILS"} validation (tolerance: 7%)`
);

// ============================================================================
// Step 9: Generate Visualizations
// ============================================================================

console.log("\n📊 STEP 9: Generating Visualizations");
console.log("─".repeat(70));

// 1. Efficient Frontier Plot
try {
  const frontierFig = new Figure({ width: 800, height: 600 });
  const frontierAx = frontierFig.addAxes();

  const frontierReturns = frontier.map((p) => p.return * 100);
  const frontierVols = frontier.map((p) => p.volatility * 100);

  frontierAx.plot(tensor(frontierVols), tensor(frontierReturns), {
    color: "#2196F3",
    linewidth: 2,
  });

  // Mark special portfolios
  frontierAx.scatter(
    tensor([minVarMetrics.volatility * 100]),
    tensor([minVarMetrics.expectedReturn * 100]),
    { color: "#4CAF50", size: 12 }
  );

  frontierAx.scatter(
    tensor([maxSharpeResult.volatility * 100]),
    tensor([maxSharpeResult.expectedReturn * 100]),
    { color: "#FF5722", size: 12 }
  );

  frontierAx.setTitle("Efficient Frontier");
  frontierAx.setXLabel("Volatility (%)");
  frontierAx.setYLabel("Expected Return (%)");

  const frontierSvg = frontierFig.renderSVG();
  writeFileSync(`${OUTPUT_DIR}/efficient-frontier.svg`, frontierSvg.svg);
  console.log(`  ✓ Saved: ${OUTPUT_DIR}/efficient-frontier.svg`);
} catch (e) {
  console.log(`  ⚠ Could not generate efficient frontier plot: ${e}`);
}

// 2. Monte Carlo Distribution Plot
try {
  const mcFig = new Figure({ width: 800, height: 600 });
  const mcAx = mcFig.addAxes();

  // Create histogram data
  const numBins = 50;
  const minReturn = Math.min(...mcResult.scenarios);
  const maxReturn = Math.max(...mcResult.scenarios);
  const binWidth = (maxReturn - minReturn) / numBins;

  const bins: number[] = [];
  const counts: number[] = [];

  for (let i = 0; i < numBins; i++) {
    const binStart = minReturn + i * binWidth;
    const binEnd = binStart + binWidth;
    const binCenter = (binStart + binEnd) / 2;
    const count = mcResult.scenarios.filter((s) => s >= binStart && s < binEnd).length;
    bins.push(binCenter * 100);
    counts.push(count);
  }

  mcAx.bar(tensor(bins), tensor(counts), { color: "#9C27B0" });
  mcAx.setTitle("Monte Carlo Return Distribution");
  mcAx.setXLabel("Return (%)");
  mcAx.setYLabel("Frequency");

  const mcSvg = mcFig.renderSVG();
  writeFileSync(`${OUTPUT_DIR}/monte-carlo-distribution.svg`, mcSvg.svg);
  console.log(`  ✓ Saved: ${OUTPUT_DIR}/monte-carlo-distribution.svg`);
} catch (e) {
  console.log(`  ⚠ Could not generate Monte Carlo plot: ${e}`);
}

// 3. Correlation Heatmap
try {
  const heatFig = new Figure({ width: 800, height: 700 });
  const heatAx = heatFig.addAxes();

  heatAx.heatmap(corrMatrix);
  heatAx.setTitle("Asset Correlation Matrix");

  const heatSvg = heatFig.renderSVG();
  writeFileSync(`${OUTPUT_DIR}/correlation-heatmap.svg`, heatSvg.svg);
  console.log(`  ✓ Saved: ${OUTPUT_DIR}/correlation-heatmap.svg`);
} catch (e) {
  console.log(`  ⚠ Could not generate correlation heatmap: ${e}`);
}

// ============================================================================
// Step 10: Final Summary
// ============================================================================

console.log(`\n${"═".repeat(70)}`);
console.log("  ANALYSIS COMPLETE - SUMMARY");
console.log("═".repeat(70));

console.log("\n📌 Key Findings:\n");
console.log("  1. Portfolio Comparison:");
console.log(`     • Equal-Weight Sharpe:   ${initialMetrics.sharpeRatio.toFixed(3)}`);
console.log(`     • Min-Variance Sharpe:   ${minVarMetrics.sharpeRatio.toFixed(3)}`);
console.log(`     • Max-Sharpe Sharpe:     ${maxSharpeResult.sharpeRatio.toFixed(3)}`);
console.log(`     • Risk-Parity Sharpe:    ${riskParityMetrics.sharpeRatio.toFixed(3)}`);

console.log("\n  2. Risk Assessment:");
console.log(`     • 95% VaR:               ${(riskMetrics.var95 * 100).toFixed(2)}%`);
console.log(`     • 95% CVaR:              ${(riskMetrics.cvar95 * 100).toFixed(2)}%`);
console.log(`     • Maximum Drawdown:      ${(riskMetrics.maxDrawdown * 100).toFixed(2)}%`);

console.log("\n  3. Monte Carlo Insights:");
console.log(`     • Expected 12M Return:   ${(mcResult.meanReturn * 100).toFixed(2)}%`);
console.log(`     • Worst Case (5%):       ${(mcResult.percentile5 * 100).toFixed(2)}%`);
console.log(`     • Best Case (95%):       ${(mcResult.percentile95 * 100).toFixed(2)}%`);

console.log("\n  4. Stress Test Worst Cases:");
const worstStress = stressResults.reduce((worst, r) =>
  r.portfolioImpact < worst.portfolioImpact ? r : worst
);
console.log(`     • ${worstStress.scenario}: ${(worstStress.portfolioImpact * 100).toFixed(2)}%`);

console.log("\n📁 Output Files:");
console.log(`   • ${OUTPUT_DIR}/efficient-frontier.svg`);
console.log(`   • ${OUTPUT_DIR}/monte-carlo-distribution.svg`);
console.log(`   • ${OUTPUT_DIR}/correlation-heatmap.svg`);

console.log(`\n${"═".repeat(70)}`);
console.log("  ✅ Financial Risk Analysis Complete!");
console.log("═".repeat(70));
