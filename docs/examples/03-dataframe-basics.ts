/**
 * Example 3: DataFrame Basics
 *
 * Learn fundamental DataFrame operations for working with tabular data.
 * Covers creation, selection, filtering, and sorting.
 * DataFrames are similar to Pandas DataFrames or spreadsheet tables.
 */

import { DataFrame } from "deepbox/dataframe";

const expectNumber = (value: unknown): number => {
  if (typeof value !== "number") {
    throw new Error("Expected number");
  }
  return value;
};

console.log("=== DataFrame Basics ===\n");

// Create a DataFrame from an object with column arrays

// Creating a DataFrame from an object
const df = new DataFrame({
  name: ["Alice", "Bob", "Charlie", "David", "Eve"],
  age: [25, 30, 35, 28, 32],
  salary: [50000, 60000, 75000, 55000, 70000],
  department: ["IT", "HR", "IT", "Sales", "HR"],
});

// Display the complete DataFrame
console.log("Full DataFrame:");
console.log(df.toString());

// Display DataFrame dimensions
console.log(`\nShape: ${df.shape[0]} rows × ${df.shape[1]} columns`);

// Access individual columns
console.log("\nColumn Access:");
console.log(`Columns: ${df.columns.join(", ")}\n`);

// Accessing columns
// Get a single column as a Series
const ages = df.get("age");
console.log("Age column:");
console.log(`${ages.toString()}\n`);

// Selecting specific columns
// Select multiple columns at once
const subset = df.select(["name", "salary"]);
console.log("\nSelecting Multiple Columns:");
console.log("Selected columns (name, salary):");
console.log(`${subset.toString()}\n`);

// Filtering rows
// Filter for high earners (salary > 60000)
const highEarners = df.filter((row) => expectNumber(row.salary) > 60000);
console.log("Employees with salary > 60000:");
console.log(`${highEarners.toString()}\n`);

// Sorting
// Sort by salary in descending order
const sortedBySalary = df.sort("salary", false);
console.log("\nSorting:");
console.log("Sorted by salary (descending):");
console.log(`${sortedBySalary.toString()}\n`);

// Head and tail
console.log("\nFiltering Rows:");
console.log("First 3 rows:");
console.log(`${df.head(3).toString()}\n`);

console.log("Last 2 rows:");
console.log(`${df.tail(2).toString()}\n`);

console.log("✓ DataFrame basics complete!");
