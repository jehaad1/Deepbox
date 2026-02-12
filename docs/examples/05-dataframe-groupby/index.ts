/**
 * Example 05: DataFrame GroupBy & Aggregation
 *
 * Learn to group and aggregate data for analysis.
 * Similar to SQL GROUP BY or Pandas groupby operations.
 */

import { DataFrame } from "deepbox/dataframe";

console.log("=== DataFrame GroupBy & Aggregation ===\n");

// Create sample sales data
const sales = new DataFrame({
  product: ["Laptop", "Mouse", "Laptop", "Keyboard", "Mouse", "Laptop", "Keyboard", "Mouse"],
  region: ["North", "North", "South", "North", "South", "North", "South", "South"],
  quantity: [5, 20, 3, 15, 25, 4, 10, 30],
  revenue: [5000, 400, 3000, 450, 500, 4000, 300, 600],
});

// Display the raw data
console.log("Sales Data:");
console.log(`${sales.toString()}\n`);

// Group by a single column and aggregate
console.log("\nGroup by single column");
// Group by product and calculate total sales
const byProduct = sales.groupBy("product");
const productStats = byProduct.agg({
  quantity: "sum",
  revenue: "sum",
});

console.log("Sales by Product:");
console.log(`${productStats.toString()}\n`);

// Group by region
// Calculate average sales by region
const byRegion = sales.groupBy("region");
const regionStats = byRegion.agg({
  quantity: "mean",
  revenue: "mean",
});

console.log("Average Sales by Region:");
console.log(`${regionStats.toString()}\n`);

// Perform multiple aggregations at once
console.log("\nMultiple aggregations");
// Calculate sum, mean, and count for each product
const detailedStats = byProduct.agg({
  quantity: "sum",
  revenue: "mean",
});

console.log("Detailed Product Statistics:");
console.log(`${detailedStats.toString()}\n`);

console.log("✓ GroupBy operations complete!");
