/**
 * Example 44: Advanced Visualization
 *
 * New in v1.0.0: Comprehensive plotting module with line plots, scatter,
 * histograms, heatmaps, confusion matrices, ROC curves, learning curves,
 * decision boundaries, feature importances, and more.
 */

import { linspace, tensor } from "deepbox/ndarray";
import {
  bar,
  heatmap,
  hist,
  plot,
  plotConfusionMatrix,
  plotElbowCurve,
  plotFeatureImportance,
  plotResiduals,
  plotRocCurve,
  scatter,
  show,
} from "deepbox/plot";

console.log("=".repeat(60));
console.log("Example 44: Advanced Visualization");
console.log("=".repeat(60));

// ============================================================================
// Part 1: Line Plot
// ============================================================================
console.log("\n📈 Part 1: Line Plot");
console.log("-".repeat(60));

// Create x values from 0 to 2π
const x = linspace(0, 2 * Math.PI, 100);
const sinData: number[] = [];
const cosData: number[] = [];
const xData = x.data as Float64Array;

for (let i = 0; i < x.size; i++) {
  const val = Number(xData[x.offset + i]);
  sinData.push(Math.sin(val));
  cosData.push(Math.cos(val));
}

const ySin = tensor(sinData);
const yCos = tensor(cosData);

// Plot sine and cosine curves
plot(x, ySin, { color: "blue", label: "sin(x)" });
plot(x, yCos, { color: "red", label: "cos(x)" });
const lineSvg = show({ format: "svg" });
console.log("Line plot (sin/cos) rendered:");
console.log(`  SVG output: ${lineSvg.svg.length} characters`);

// ============================================================================
// Part 2: Scatter Plot
// ============================================================================
console.log("\n🟠 Part 2: Scatter Plot");
console.log("-".repeat(60));

const scatterX = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const scatterY = tensor([2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.8, 16.1, 18.2, 19.9]);

scatter(scatterX, scatterY, { color: "blue", label: "data points" });
const scatterSvg = show({ format: "svg" });
console.log("Scatter plot rendered:");
console.log(`  SVG output: ${scatterSvg.svg.length} characters`);

// ============================================================================
// Part 3: Bar Chart
// ============================================================================
console.log("\n📊 Part 3: Bar Chart");
console.log("-".repeat(60));

const categories = tensor([1, 2, 3, 4, 5]);
const values = tensor([23, 45, 12, 67, 34]);

bar(categories, values, { color: "steelblue", label: "Sales" });
const barSvg = show({ format: "svg" });
console.log("Bar chart rendered:");
console.log(`  SVG output: ${barSvg.svg.length} characters`);

// ============================================================================
// Part 4: Histogram
// ============================================================================
console.log("\n📉 Part 4: Histogram");
console.log("-".repeat(60));

// Generate random-like data
const histData: number[] = [];
for (let i = 0; i < 200; i++) {
  // Box-Muller approximation for normal distribution
  const u1 = (i + 1) / 201;
  const u2 = (((i * 7 + 3) % 200) + 1) / 201;
  histData.push(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
}

hist(tensor(histData), 20, { color: "green", label: "Normal-like data" });
const histSvg = show({ format: "svg" });
console.log("Histogram (20 bins) rendered:");
console.log(`  SVG output: ${histSvg.svg.length} characters`);

// ============================================================================
// Part 5: Heatmap
// ============================================================================
console.log("\n🟥 Part 5: Heatmap");
console.log("-".repeat(60));

const heatmapData = tensor([
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
]);

heatmap(heatmapData, { label: "4×4 heatmap" });
const heatSvg = show({ format: "svg" });
console.log("Heatmap rendered:");
console.log(`  SVG output: ${heatSvg.svg.length} characters`);

// ============================================================================
// Part 6: Confusion Matrix
// ============================================================================
console.log("\n🎯 Part 6: Confusion Matrix");
console.log("-".repeat(60));

const confMatrix = tensor([
  [45, 5, 2],
  [3, 40, 7],
  [1, 4, 43],
]);

plotConfusionMatrix(confMatrix, ["Cat", "Dog", "Bird"]);
const cmSvg = show({ format: "svg" });
console.log("Confusion Matrix rendered:");
console.log(`  SVG output: ${cmSvg.svg.length} characters`);
console.log("  Classes: Cat, Dog, Bird");

// ============================================================================
// Part 7: ROC Curve
// ============================================================================
console.log("\n📐 Part 7: ROC Curve");
console.log("-".repeat(60));

// Simulated ROC curve data
const fpr = tensor([0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]);
const tpr = tensor([0, 0.4, 0.65, 0.8, 0.88, 0.94, 0.98, 1.0]);

plotRocCurve(fpr, tpr, 0.87);
const rocSvg = show({ format: "svg" });
console.log("ROC Curve rendered (AUC = 0.87):");
console.log(`  SVG output: ${rocSvg.svg.length} characters`);

// ============================================================================
// Part 8: Feature Importance
// ============================================================================
console.log("\n⭐ Part 8: Feature Importance");
console.log("-".repeat(60));

const importances = tensor([0.35, 0.25, 0.15, 0.12, 0.08, 0.05]);
const featureNames = ["income", "age", "credit_score", "tenure", "balance", "products"];

plotFeatureImportance(importances, featureNames);
const fiSvg = show({ format: "svg" });
console.log("Feature Importance chart rendered:");
console.log(`  SVG output: ${fiSvg.svg.length} characters`);

// ============================================================================
// Part 9: Elbow Curve
// ============================================================================
console.log("\n📏 Part 9: Elbow Curve (KMeans)");
console.log("-".repeat(60));

const kValues = tensor([2, 3, 4, 5, 6, 7, 8]);
const inertias = tensor([500, 300, 180, 120, 100, 90, 85]);

plotElbowCurve(kValues, inertias);
const elbowSvg = show({ format: "svg" });
console.log("Elbow Curve rendered:");
console.log(`  SVG output: ${elbowSvg.svg.length} characters`);
console.log("  Optimal k appears around 4-5 (elbow point)");

// ============================================================================
// Part 10: Residual Plot
// ============================================================================
console.log("\n📊 Part 10: Residual Plot");
console.log("-".repeat(60));

const yTrue = tensor([3, 5, 7, 9, 11, 13, 15]);
const yPred = tensor([3.1, 4.8, 7.3, 8.7, 11.2, 12.8, 15.1]);

plotResiduals(yTrue, yPred);
const resSvg = show({ format: "svg" });
console.log("Residual Plot rendered:");
console.log(`  SVG output: ${resSvg.svg.length} characters`);

// ============================================================================
// Summary
// ============================================================================
console.log("\n💡 Key Takeaways");
console.log("-".repeat(60));
console.log("• plot/scatter: line and point plots for continuous data");
console.log("• bar/barh: vertical and horizontal bar charts for categorical data");
console.log("• hist: histograms for distribution visualization");
console.log("• heatmap: 2D color-coded matrix visualization");
console.log("• plotConfusionMatrix: classification performance at a glance");
console.log("• plotRocCurve: binary classifier threshold tradeoffs");
console.log("• plotFeatureImportance: which features matter most");
console.log("• plotElbowCurve: optimal number of clusters (KMeans)");
console.log("• plotResiduals: regression model diagnostics");
console.log("• All plots render to SVG (vector) or PNG (raster)");

console.log("\n✅ Advanced Visualization Example Complete!");
console.log("=".repeat(60));
