/**
 * Example 19: Statistics
 *
 * Descriptive statistics, percentiles and correlation on small samples.
 * The statistics functions return tensors, and item() turns a one-element
 * tensor into a number. See example 43 for hypothesis tests.
 */

import { stack, tensor } from "deepbox/ndarray";
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

console.log("=== Statistics ===\n");

// Sample data
const data = tensor([23, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 55, 60, 65]);

console.log("Dataset:");
console.log(`${data.toString()}\n`);

// Descriptive statistics
console.log("Descriptive Statistics:");
console.log("-".repeat(50));

const meanVal = Number(mean(data).item());
const medianVal = Number(median(data).item());
const stdVal = Number(std(data).item());
const varVal = Number(variance(data).item());
const skewVal = Number(skewness(data).item());
const kurtVal = Number(kurtosis(data).item());

console.log(`Mean:     ${meanVal.toFixed(2)}`);
console.log(`Median:   ${medianVal.toFixed(2)}`);
console.log(`Std Dev:  ${stdVal.toFixed(2)}`);
console.log(`Variance: ${varVal.toFixed(2)}`);
console.log(`Skewness: ${skewVal.toFixed(4)}`);
console.log(`Kurtosis: ${kurtVal.toFixed(4)} (excess kurtosis, 0 for a normal distribution)\n`);

// std and variance divide by n by default (the population value, like NumPy).
// Pass ddof: 1 for the sample value (like pandas).
const sampleStd = Number(std(data, { ddof: 1 }).item());
console.log(`Sample Std Dev (ddof = 1): ${sampleStd.toFixed(2)}\n`);

// Percentiles
console.log("Percentiles:");
console.log("-".repeat(50));

const p25 = Number(percentile(data, 25).item());
const p50 = Number(percentile(data, 50).item());
const p75 = Number(percentile(data, 75).item());

console.log(`25th percentile (Q1): ${p25.toFixed(2)}`);
console.log(`50th percentile (Q2): ${p50.toFixed(2)}`);
console.log(`75th percentile (Q3): ${p75.toFixed(2)}\n`);

// Several percentiles at once return one tensor.
console.log(`Quartiles in one call: ${percentile(data, [25, 50, 75]).toString()}\n`);

// Correlation analysis
console.log("Correlation Analysis:");
console.log("-".repeat(50));

const x = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const y = tensor([2.1, 4.2, 5.8, 8.1, 10.3, 11.9, 14.2, 16.1, 17.8, 20.2]);

// pearsonr returns [coefficient, p-value]. The default test is two-sided.
const [r, pTwoSided] = pearsonr(x, y);
console.log(`Pearson correlation: ${r.toFixed(4)} (p = ${pTwoSided.toExponential(2)})`);

// Test one direction only: the alternative is that the correlation is greater than zero.
const [, pGreater] = pearsonr(x, y, { alternative: "greater" });
console.log(`One-sided p-value (alternative "greater"): ${pGreater.toExponential(2)}`);

// corrcoef on a matrix: rows are observations and columns are variables.
// stack(..., 1) joins three vectors as the columns x, y and z, where z falls as x rises.
const z = tensor([9, 8, 8.5, 6, 5.5, 5, 3, 3.5, 1, 0.5]);
const observations = stack([x, y, z], 1);
console.log(`\nCorrelation Matrix (columns x, y, z, shape [${observations.shape}]):`);
console.log(corrcoef(observations).toString());
