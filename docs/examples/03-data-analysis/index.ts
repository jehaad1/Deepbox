/**
 * Example 03: Data Analysis & Visualization
 *
 * An exploratory workflow on a small employee table: summarize it with a
 * DataFrame, compute statistics, group and filter rows, and draw four charts.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { DataFrame } from "deepbox/dataframe";
import { tensor } from "deepbox/ndarray";
import { Figure } from "deepbox/plot";
import { corrcoef, mean, std } from "deepbox/stats";

const expectNumber = (value: unknown): number => {
  if (typeof value !== "number") {
    throw new Error("Expected number");
  }
  return value;
};

const expectNumberArray = (value: unknown): number[] => {
  if (!Array.isArray(value) || value.some((v) => typeof v !== "number")) {
    throw new Error("Expected number[]");
  }
  return value;
};

console.log("=".repeat(60));
console.log("Example 03: Data Analysis & Visualization");
console.log("=".repeat(60));

mkdirSync("docs/examples/03-data-analysis/output", { recursive: true });

// Create a DataFrame with employee information
const employeeData = new DataFrame({
  name: [
    "Alice",
    "Bob",
    "Charlie",
    "David",
    "Eve",
    "Frank",
    "Grace",
    "Henry",
    "Ivy",
    "Jack",
    "Kate",
    "Leo",
    "Mia",
    "Noah",
    "Olivia",
    "Paul",
    "Quinn",
    "Rachel",
    "Sam",
    "Tina",
  ],
  department: [
    "Engineering",
    "Sales",
    "Engineering",
    "HR",
    "Engineering",
    "Sales",
    "Marketing",
    "Engineering",
    "HR",
    "Sales",
    "Engineering",
    "Marketing",
    "Sales",
    "Engineering",
    "HR",
    "Sales",
    "Engineering",
    "Marketing",
    "Engineering",
    "Sales",
  ],
  salary: [
    95000, 65000, 105000, 55000, 98000, 72000, 68000, 110000, 58000, 70000, 102000, 71000, 67000,
    115000, 60000, 69000, 108000, 73000, 112000, 66000,
  ],
  experience: [5, 3, 8, 2, 6, 4, 3, 10, 2, 5, 7, 4, 3, 12, 3, 4, 9, 5, 11, 3],
  age: [28, 25, 32, 24, 30, 27, 26, 35, 24, 29, 31, 28, 26, 38, 27, 28, 34, 30, 36, 26],
});

// Display dataset overview
console.log("\nDataset Overview");
console.log("-".repeat(60));
console.log(`Total Employees: ${employeeData.shape[0]}`);
console.log(`Columns: ${employeeData.columns.join(", ")}`);

// Show first few rows
console.log("\nFirst 5 Rows:");
console.log(employeeData.head(5).toString());

// Calculate descriptive statistics
console.log("\nStatistical Summary");
console.log("-".repeat(60));

// Extract columns as arrays for analysis
const salaries = expectNumberArray(employeeData.get("salary").toArray());
const experiences = expectNumberArray(employeeData.get("experience").toArray());
const ages = expectNumberArray(employeeData.get("age").toArray());

// Convert arrays to tensors for statistical operations
const salaryTensor = tensor(salaries);
const expTensor = tensor(experiences);

// Calculate salary statistics
const salaryMean = Number(mean(salaryTensor).item());
const salarySd = Number(std(salaryTensor).item());

console.log(`Salary Statistics:`);
console.log(`  Mean: $${salaryMean.toFixed(2)}`);
console.log(`  Std Dev: $${salarySd.toFixed(2)}`);
console.log(`  Min: $${Math.min(...salaries)}`);
console.log(`  Max: $${Math.max(...salaries)}`);

// Calculate experience statistics
const expMean = Number(mean(expTensor).item());
const expSd = Number(std(expTensor).item());

console.log(`\nExperience Statistics:`);
console.log(`  Mean: ${expMean.toFixed(1)} years`);
console.log(`  Std Dev: ${expSd.toFixed(1)} years`);

// Group by department and calculate averages
console.log("\nDepartment Analysis");
console.log("-".repeat(60));

// GroupBy operation to aggregate by department
const deptGroups = employeeData.groupBy("department");
const deptStats = deptGroups.agg({
  salary: "mean",
  experience: "mean",
});

console.log("Average Salary by Department:");
console.log(deptStats.toString());

// Filter data based on conditions
console.log("\nFiltering Examples");
console.log("-".repeat(60));

// Find employees earning over $100k
const highEarners = employeeData.filter((row) => expectNumber(row.salary) > 100000);
console.log(`High Earners (>$100k): ${highEarners.shape[0]} employees`);
console.log(highEarners.select(["name", "department", "salary"]).toString());

// Filter by department
const engineeringDept = employeeData.filter((row) => row.department === "Engineering");
console.log(`\nEngineering Department: ${engineeringDept.shape[0]} employees`);

// Analyze correlations between variables
console.log("\nCorrelation Analysis");
console.log("-".repeat(60));

// corrcoef expects rows=observations and columns=variables.
const dataMatrix = tensor(
  salaries.map((salary, index) => [salary, experiences[index] ?? 0, ages[index] ?? 0])
);
const correlationMatrix = corrcoef(dataMatrix);

console.log("Correlation Matrix (Salary, Experience, Age):");
console.log(correlationMatrix.toString());

// Generate visualizations
console.log("\nCreating Visualizations");
console.log("-".repeat(60));

// 1. Scatter plot showing relationship between experience and salary
console.log("1. Scatter Plot: Salary vs Experience");
const fig1 = new Figure();
const ax1 = fig1.addAxes();
ax1.scatter(expTensor, salaryTensor, { color: "#1f77b4", size: 8 });
ax1.setTitle("Salary vs Experience");
ax1.setXLabel("Years of Experience");
ax1.setYLabel("Salary ($)");
const svg1 = fig1.renderSVG();
writeFileSync("docs/examples/03-data-analysis/output/salary-vs-experience.svg", svg1.svg);
console.log("   Saved: output/salary-vs-experience.svg");

// 2. Histogram showing salary distribution
console.log("2. Histogram: Salary Distribution");
const fig2 = new Figure();
const ax2 = fig2.addAxes();
ax2.hist(salaryTensor, 8, { color: "#2ca02c" });
ax2.setTitle("Salary Distribution");
ax2.setXLabel("Salary ($)");
ax2.setYLabel("Frequency");
const svg2 = fig2.renderSVG();
writeFileSync("docs/examples/03-data-analysis/output/salary-distribution.svg", svg2.svg);
console.log("   Saved: output/salary-distribution.svg");

// 3. Bar chart comparing departments
console.log("3. Bar Chart: Average Salary by Department");
// Calculate average salary for each department
const depts = ["Engineering", "Sales", "Marketing", "HR"];
const avgSalaries = depts.map((dept) => {
  const deptData = employeeData.filter((row) => row.department === dept);
  const deptSalaries = expectNumberArray(deptData.get("salary").toArray());
  return Number(mean(tensor(deptSalaries)).item());
});

const fig3 = new Figure();
const ax3 = fig3.addAxes();
ax3.bar(tensor([0, 1, 2, 3]), tensor(avgSalaries), {
  color: "#ff7f0e",
  edgecolor: "#000000",
});
ax3.setTitle("Average Salary by Department");
ax3.setXLabel("Department");
ax3.setYLabel("Average Salary ($)");
const svg3 = fig3.renderSVG();
writeFileSync("docs/examples/03-data-analysis/output/dept-salaries.svg", svg3.svg);
console.log("   Saved: output/dept-salaries.svg");

// 4. Heatmap visualizing correlations
console.log("4. Heatmap: Correlation Matrix");
const fig4 = new Figure();
const ax4 = fig4.addAxes();
ax4.heatmap(correlationMatrix, { vmin: -1, vmax: 1 });
ax4.setTitle("Correlation Matrix");
const svg4 = fig4.renderSVG();
writeFileSync("docs/examples/03-data-analysis/output/correlation-heatmap.svg", svg4.svg);
console.log("   Saved: output/correlation-heatmap.svg");

// Summary of findings, computed from the data above
console.log("\nFindings");
console.log("-".repeat(60));
const bestIndex = avgSalaries.indexOf(Math.max(...avgSalaries));
console.log(`- ${depts[bestIndex]} has the highest average salary`);
const salaryExperienceR = correlationMatrix.at(0, 1);
const salaryAgeR = correlationMatrix.at(0, 2);
console.log(`- Salary vs experience correlation: ${Number(salaryExperienceR).toFixed(2)}`);
console.log(`- Salary vs age correlation: ${Number(salaryAgeR).toFixed(2)}`);

console.log("\nAnalysis complete.");
console.log("=".repeat(60));
