/**
 * Example 05: DataFrame GroupBy & Aggregation
 *
 * Group rows by one or more columns and aggregate each group, like SQL GROUP BY.
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

// Group by one column and sum two others
console.log("Group by a single column");
const byProduct = sales.groupBy("product");
const productStats = byProduct.agg({
  quantity: "sum",
  revenue: "sum",
});

console.log("Sales by Product:");
console.log(`${productStats.toString()}\n`);

// Group by region and take the mean of each numeric column
const byRegion = sales.groupBy("region");
const regionStats = byRegion.agg({
  quantity: "mean",
  revenue: "mean",
});

console.log("Average Sales by Region:");
console.log(`${regionStats.toString()}\n`);

// Use a different function per column
console.log("Different function per column");
const detailedStats = byProduct.agg({
  quantity: "sum",
  revenue: "mean",
});

console.log("Total quantity and mean revenue by product:");
console.log(`${detailedStats.toString()}\n`);

// Named aggregation: the key is the output column, the value is [column, function]
console.log("Named aggregation");
const named = byProduct.agg({
  unitsSold: ["quantity", "sum"],
  avgRevenue: ["revenue", "mean"],
  orders: ["revenue", "count"],
});
console.log(`${named.toString()}\n`);

// Group by two columns
console.log("Group by two columns");
console.log(`${sales.groupBy(["product", "region"]).agg({ revenue: "sum" }).toString()}\n`);

// Pull out the rows of a single group
console.log("Rows of the Laptop group:");
console.log(`${byProduct.getGroup("Laptop").toString()}\n`);

// Number of distinct values per group
console.log("Distinct values per product:");
console.log(byProduct.nunique().toString());
