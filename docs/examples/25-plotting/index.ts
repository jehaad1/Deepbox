/**
 * Example 25: Data Visualization
 *
 * Create various types of plots to visualize data and results.
 * Deepbox supports SVG and PNG output for publication-quality figures.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { cos, linspace, sin, tensor } from "deepbox/ndarray";
import { Figure } from "deepbox/plot";

console.log("=== Data Visualization ===\n");

mkdirSync("docs/examples/25-plotting/output", { recursive: true });

// 1. Line Plot
console.log("1. Creating line plot...");
const x = linspace(0, 2 * Math.PI, 100);
const y1 = sin(x);
const y2 = cos(x);

const fig1 = new Figure({ width: 640, height: 480 });
const ax1 = fig1.addAxes();
ax1.plot(x, y1, { color: "#1f77b4", linewidth: 2 });
ax1.plot(x, y2, { color: "#ff7f0e", linewidth: 2 });
ax1.setTitle("Sine and Cosine Functions");
ax1.setXLabel("x");
ax1.setYLabel("y");

const svg1 = fig1.renderSVG();
writeFileSync("docs/examples/25-plotting/output/line-plot.svg", svg1.svg);
console.log("   ✓ Saved: output/line-plot.svg\n");

// 2. Scatter Plot
console.log("2. Creating scatter plot...");
const x_scatter = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const y_scatter = tensor([2.3, 4.1, 5.9, 7.8, 10.2, 11.8, 14.3, 16.1, 17.9, 20.1]);

const fig2 = new Figure();
const ax2 = fig2.addAxes();
ax2.scatter(x_scatter, y_scatter, { color: "#2ca02c", size: 8 });
ax2.setTitle("Scatter Plot Example");
ax2.setXLabel("X values");
ax2.setYLabel("Y values");

const svg2 = fig2.renderSVG();
writeFileSync("docs/examples/25-plotting/output/scatter-plot.svg", svg2.svg);
console.log("   ✓ Saved: output/scatter-plot.svg\n");

// 3. Bar Chart
console.log("3. Creating bar chart...");
const categories = tensor([0, 1, 2, 3, 4]);
const values = tensor([23, 45, 56, 78, 32]);

const fig3 = new Figure();
const ax3 = fig3.addAxes();
ax3.bar(categories, values, { color: "#d62728", edgecolor: "#000000" });
ax3.setTitle("Bar Chart Example");
ax3.setXLabel("Categories");
ax3.setYLabel("Values");

const svg3 = fig3.renderSVG();
writeFileSync("docs/examples/25-plotting/output/bar-chart.svg", svg3.svg);
console.log("   ✓ Saved: output/bar-chart.svg\n");

// 4. Histogram
console.log("4. Creating histogram...");
const data = tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9]);

const fig4 = new Figure();
const ax4 = fig4.addAxes();
ax4.hist(data, 9, { color: "#9467bd" });
ax4.setTitle("Histogram Example");
ax4.setXLabel("Value");
ax4.setYLabel("Frequency");

const svg4 = fig4.renderSVG();
writeFileSync("docs/examples/25-plotting/output/histogram.svg", svg4.svg);
console.log("   ✓ Saved: output/histogram.svg\n");

// 5. Heatmap
console.log("5. Creating heatmap...");
const heatmap_data = tensor([
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
]);

const fig5 = new Figure();
const ax5 = fig5.addAxes();
ax5.heatmap(heatmap_data, { vmin: 1, vmax: 12 });
ax5.setTitle("Heatmap Example");

const svg5 = fig5.renderSVG();
writeFileSync("docs/examples/25-plotting/output/heatmap.svg", svg5.svg);
console.log("   ✓ Saved: output/heatmap.svg\n");

console.log("Visualization Tips:");
console.log("• Use line plots for continuous data");
console.log("• Use scatter plots to show relationships");
console.log("• Use bar charts for categorical comparisons");
console.log("• Use histograms for distributions");
console.log("• Use heatmaps for matrix data");

console.log("\n✓ Plotting complete!");
