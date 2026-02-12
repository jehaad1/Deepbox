/**
 * Example 19: Statistical Analysis
 *
 * Perform descriptive statistics and hypothesis testing.
 * Learn to analyze data distributions and test hypotheses.
 */

import { tensor } from "deepbox/ndarray";
import {
  corrcoef,
  kurtosis,
  mean,
  median,
  pearsonr,
  percentile,
  skewness,
  std,
  variance,
} from "deepbox/stats";

console.log("=== Statistical Analysis ===\n");

// Sample data
const data = tensor([23, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 55, 60, 65]);

console.log("Dataset:");
console.log(`${data.toString()}\n`);

// Descriptive statistics
console.log("Descriptive Statistics:");
console.log("-".repeat(50));

const meanVal = Number(mean(data).data[0]);
const medianVal = Number(median(data).data[0]);
const stdVal = Number(std(data).data[0]);
const varVal = Number(variance(data).data[0]);
const skewVal = Number(skewness(data).data[0]);
const kurtVal = Number(kurtosis(data).data[0]);

console.log(`Mean:     ${meanVal.toFixed(2)}`);
console.log(`Median:   ${medianVal.toFixed(2)}`);
console.log(`Std Dev:  ${stdVal.toFixed(2)}`);
console.log(`Variance: ${varVal.toFixed(2)}`);
console.log(`Skewness: ${skewVal.toFixed(4)}`);
console.log(`Kurtosis: ${kurtVal.toFixed(4)}\n`);

// Percentiles
console.log("Percentiles:");
console.log("-".repeat(50));

const p25 = Number(percentile(data, 25).data[0]);
const p50 = Number(percentile(data, 50).data[0]);
const p75 = Number(percentile(data, 75).data[0]);

console.log(`25th percentile (Q1): ${p25.toFixed(2)}`);
console.log(`50th percentile (Q2): ${p50.toFixed(2)}`);
console.log(`75th percentile (Q3): ${p75.toFixed(2)}\n`);

// Correlation analysis
console.log("Correlation Analysis:");
console.log("-".repeat(50));

const x = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const y = tensor([2.1, 4.2, 5.8, 8.1, 10.3, 11.9, 14.2, 16.1, 17.8, 20.2]);

const correlationResult = pearsonr(x, y);
const correlation = Array.isArray(correlationResult) ? correlationResult[0] : correlationResult;
console.log(`Pearson correlation: ${Number(correlation).toFixed(4)}`);

const corrMatrix = corrcoef(
  tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
  ])
);
console.log("\nCorrelation Matrix:");
console.log(corrMatrix.toString());

console.log("\n✓ Statistical analysis complete!");
