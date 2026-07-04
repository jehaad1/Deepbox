/**
 * Example 33: Advanced DataFrame Features
 *
 * New in v1.0.0: String and DateTime accessors, rolling/expanding/EWM windows,
 * query/eval expressions, pivot tables, crosstabs, interpolation, and more.
 */

import { DataFrame, Series } from "deepbox/dataframe";

console.log("=".repeat(60));
console.log("Example 33: Advanced DataFrame Features");
console.log("=".repeat(60));

// ============================================================================
// Part 1: Series String Accessor (str)
// ============================================================================
console.log("\n📝 Part 1: Series String Accessor");
console.log("-".repeat(60));

// The .str accessor provides vectorized string operations on Series
const names = new Series(["Alice Smith", "bob jones", "CHARLIE BROWN", null, "diana prince"]);

console.log("Original names:");
console.log(names.toString());

// Case transformations
const upper = names.str.upper();
console.log("\n.str.upper():", upper.toString());

const lower = names.str.lower();
console.log(".str.lower():", lower.toString());

const title = names.str.title();
console.log(".str.title():", title.toString());

// String matching
const containsI = names.str.contains("i");
console.log("\n.str.contains('i'):", containsI.toString());

// String operations
const lengths = names.str.len();
console.log(".str.len():", lengths.toString());

const replaced = names.str.replace("o", "0");
console.log(".str.replace('o', '0'):", replaced.toString());

const trimmed = new Series(["  hello  ", " world ", null]);
console.log("\n.str.strip():", trimmed.str.strip().toString());

// Split and slice
const emails = new Series(["alice@example.com", "bob@test.org", null]);
console.log(".str.split('@'):", emails.str.split("@").toString());

const starts = names.str.startswith("A");
console.log(".str.startswith('A'):", starts.toString());

const ends = names.str.endswith("e");
console.log(".str.endswith('e'):", ends.toString());

// ============================================================================
// Part 2: Series DateTime Accessor (dt)
// ============================================================================
console.log("\n📅 Part 2: Series DateTime Accessor");
console.log("-".repeat(60));

// The .dt accessor provides vectorized datetime extraction on Series
const dates = new Series([
  new Date("2024-01-15T10:30:00"),
  new Date("2024-06-20T14:45:30"),
  new Date("2024-12-25T00:00:00"),
  null,
  new Date("2024-03-08T08:15:00"),
]);

console.log("Original dates:");
console.log(dates.toString());

// Extract date components
const years = dates.dt.year();
console.log("\n.dt.year():", years.toString());

const months = dates.dt.month();
console.log(".dt.month():", months.toString());

const days = dates.dt.day();
console.log(".dt.day():", days.toString());

const hours = dates.dt.hour();
console.log(".dt.hour():", hours.toString());

const dayOfWeek = dates.dt.dayofweek();
console.log(".dt.dayofweek():", dayOfWeek.toString());

const quarter = dates.dt.quarter();
console.log(".dt.quarter():", quarter.toString());

// ============================================================================
// Part 3: Rolling Window Calculations
// ============================================================================
console.log("\n📊 Part 3: Rolling Window Calculations");
console.log("-".repeat(60));

// Rolling windows compute statistics over a sliding window
const stockPrices = new DataFrame({
  price: [100, 102, 101, 105, 108, 107, 110, 112, 109, 115],
  volume: [1000, 1200, 900, 1500, 1300, 1100, 1400, 1600, 1000, 1800],
});

console.log("Stock price data:");
console.log(stockPrices.toString());

// Rolling mean with window size 3
const rollingMean = stockPrices.rolling(3).mean();
console.log("\nRolling mean (window=3):");
console.log(rollingMean.toString());

// Rolling standard deviation
const rollingStd = stockPrices.rolling(3).std();
console.log("Rolling std (window=3):");
console.log(rollingStd.toString());

// Rolling sum
const rollingSum = stockPrices.rolling(3).sum();
console.log("Rolling sum (window=3):");
console.log(rollingSum.toString());

// Rolling min/max
const rollingMin = stockPrices.rolling(3).min();
console.log("Rolling min (window=3):");
console.log(rollingMin.toString());

const rollingMax = stockPrices.rolling(3).max();
console.log("Rolling max (window=3):");
console.log(rollingMax.toString());

// ============================================================================
// Part 4: Expanding Window Calculations
// ============================================================================
console.log("\n📈 Part 4: Expanding Window Calculations");
console.log("-".repeat(60));

// Expanding windows compute cumulative statistics from the start
const values = new DataFrame({
  revenue: [10, 20, 15, 25, 30, 18, 35, 40],
});

const expandingMean = values.expanding().mean();
console.log("Expanding mean (cumulative average):");
console.log(expandingMean.toString());

const expandingSum = values.expanding().sum();
console.log("Expanding sum (cumulative sum):");
console.log(expandingSum.toString());

const expandingStd = values.expanding().std();
console.log("Expanding std (cumulative std):");
console.log(expandingStd.toString());

// ============================================================================
// Part 5: Exponentially Weighted Moving (EWM) Calculations
// ============================================================================
console.log("\n⚡ Part 5: Exponentially Weighted Moving (EWM)");
console.log("-".repeat(60));

// EWM gives more weight to recent observations
const temperatures = new DataFrame({
  temp: [20, 22, 21, 25, 24, 23, 26, 28, 27, 30],
});

// Using span parameter (alpha = 2 / (span + 1))
const ewmMean = temperatures.ewm({ span: 3 }).mean();
console.log("EWM mean (span=3):");
console.log(ewmMean.toString());

// Using direct alpha parameter
const ewmMeanAlpha = temperatures.ewm({ alpha: 0.3 }).mean();
console.log("EWM mean (alpha=0.3):");
console.log(ewmMeanAlpha.toString());

// EWM standard deviation
const ewmStd = temperatures.ewm({ span: 3 }).std();
console.log("EWM std (span=3):");
console.log(ewmStd.toString());

// ============================================================================
// Part 6: Query Expressions
// ============================================================================
console.log("\n🔍 Part 6: Query Expressions");
console.log("-".repeat(60));

// Query filters rows using string expressions
const employees = new DataFrame({
  name: ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
  department: ["Engineering", "Sales", "Engineering", "Marketing", "Sales", "Engineering"],
  salary: [95000, 65000, 88000, 72000, 61000, 105000],
  experience: [5, 3, 4, 6, 2, 8],
});

console.log("Employee data:");
console.log(employees.toString());

// Simple query
const highEarners = employees.query("salary > 80000");
console.log("\nquery('salary > 80000'):");
console.log(highEarners.toString());

// Compound query with AND
const seniorHighEarners = employees.query("salary > 70000 and experience > 4");
console.log("query('salary > 70000 and experience > 4'):");
console.log(seniorHighEarners.toString());

// Query with OR
const salesOrMarketing = employees.query("department == Sales or department == Marketing");
console.log("query('department == Sales or department == Marketing'):");
console.log(salesOrMarketing.toString());

// ============================================================================
// Part 7: Eval Expressions
// ============================================================================
console.log("\n🧮 Part 7: Eval Expressions");
console.log("-".repeat(60));

// Eval creates computed columns or filters using expressions
const metrics = new DataFrame({
  a: [1, 2, 3, 4, 5],
  b: [10, 20, 30, 40, 50],
});

console.log("Original:");
console.log(metrics.toString());

// Create a new column with eval
const withSum = metrics.eval("c = a + b");
console.log("\neval('c = a + b'):");
console.log(withSum.toString());

// Multiplication
const withProduct = metrics.eval("product = a * b");
console.log("eval('product = a * b'):");
console.log(withProduct.toString());

// Filter with eval
const filtered = metrics.eval("a > 2");
console.log("eval('a > 2') — filters rows:");
console.log(filtered.toString());

// ============================================================================
// Part 8: Assign (Functional Column Creation)
// ============================================================================
console.log("\n🔧 Part 8: Assign");
console.log("-".repeat(60));

// Assign creates new columns from arrays or functions
const sales = new DataFrame({
  product: ["Widget", "Gadget", "Doohickey", "Thingamajig"],
  price: [10, 25, 15, 30],
  quantity: [100, 50, 80, 30],
});

console.log("Original sales:");
console.log(sales.toString());

// Assign with a function
const withRevenue = sales.assign({
  revenue: (row: Record<string, unknown>) => {
    const price = Number(row["price"] ?? 0);
    const qty = Number(row["quantity"] ?? 0);
    return price * qty;
  },
  discount: [0.1, 0.05, 0.15, 0.0],
});

console.log("\nWith assigned columns (revenue, discount):");
console.log(withRevenue.toString());

// ============================================================================
// Part 9: Pivot Table
// ============================================================================
console.log("\n📋 Part 9: Pivot Table");
console.log("-".repeat(60));

// Pivot tables reshape data for cross-tabulation analysis
const salesData = new DataFrame({
  region: ["North", "South", "North", "South", "North", "South", "North", "South"],
  product: ["A", "A", "B", "B", "A", "A", "B", "B"],
  revenue: [100, 150, 200, 120, 130, 160, 180, 140],
});

console.log("Sales data:");
console.log(salesData.toString());

// Pivot: regions as rows, products as columns, mean revenue as values
const pivoted = salesData.pivot_table({
  index: "region",
  columns: "product",
  values: "revenue",
  aggFunc: "mean",
});

console.log("\nPivot table (mean revenue by region × product):");
console.log(pivoted.toString());

// Pivot with sum aggregation
const pivotedSum = salesData.pivot_table({
  index: "region",
  columns: "product",
  values: "revenue",
  aggFunc: "sum",
});

console.log("Pivot table (sum revenue by region × product):");
console.log(pivotedSum.toString());

// ============================================================================
// Part 10: Crosstab
// ============================================================================
console.log("\n📊 Part 10: Crosstab");
console.log("-".repeat(60));

// Crosstab computes frequency tables between two categorical columns
const survey = new DataFrame({
  gender: ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
  preference: ["A", "B", "A", "A", "B", "B", "A", "A", "B", "B"],
});

console.log("Survey data:");
console.log(survey.toString());

const crossResult = survey.crosstab("gender", "preference");
console.log("\nCrosstab (gender × preference):");
console.log(crossResult.toString());

// ============================================================================
// Part 11: nlargest / nsmallest
// ============================================================================
console.log("\n🏆 Part 11: nlargest / nsmallest");
console.log("-".repeat(60));

const scores = new DataFrame({
  student: ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"],
  score: [92, 85, 78, 95, 88, 73, 91],
  grade: ["A", "B", "C", "A", "B", "C", "A"],
});

console.log("Student scores:");
console.log(scores.toString());

const top3 = scores.nlargest(3, "score");
console.log("\nTop 3 scores:");
console.log(top3.toString());

const bottom3 = scores.nsmallest(3, "score");
console.log("Bottom 3 scores:");
console.log(bottom3.toString());

// ============================================================================
// Part 12: Interpolation
// ============================================================================
console.log("\n🔗 Part 12: Interpolation");
console.log("-".repeat(60));

// Interpolate fills missing values using linear or nearest interpolation
const sensorData = new DataFrame({
  time: [0, 1, 2, 3, 4, 5, 6, 7],
  reading: [10, null, null, 25, 30, null, 42, 50],
});

console.log("Sensor data with gaps:");
console.log(sensorData.toString());

const linearInterp = sensorData.interpolate("linear");
console.log("\nLinear interpolation:");
console.log(linearInterp.toString());

const nearestInterp = sensorData.interpolate("nearest");
console.log("Nearest interpolation:");
console.log(nearestInterp.toString());

// ============================================================================
// Summary
// ============================================================================
console.log("\n💡 Key Takeaways");
console.log("-".repeat(60));
console.log("• .str accessor: vectorized string ops (upper, lower, contains, split, etc.)");
console.log("• .dt accessor: datetime extraction (year, month, day, hour, dayofweek, quarter)");
console.log("• rolling(n): sliding window stats (mean, sum, std, var, min, max)");
console.log("• expanding(): cumulative stats from the start of the data");
console.log("• ewm({span}): exponentially weighted moving averages");
console.log("• query(): SQL-like row filtering with expressions");
console.log("• eval(): computed columns and expression-based filtering");
console.log("• assign(): functional column creation with arrays or functions");
console.log("• pivot_table(): reshape data for cross-tabulation analysis");
console.log("• crosstab(): frequency tables between categorical columns");
console.log("• nlargest/nsmallest: top/bottom N rows by column");
console.log("• interpolate(): fill missing values with linear or nearest method");

console.log("\n✅ Advanced DataFrame Features Example Complete!");
console.log("=".repeat(60));
