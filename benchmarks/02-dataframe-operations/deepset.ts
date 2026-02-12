/**
 * Benchmark 02: DataFrame Operations — Deepbox vs Pandas
 *
 * Compares tabular data operations: creation, selection,
 * filtering, groupby, sorting, and aggregation.
 */

import { DataFrame } from "deepbox/dataframe";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("dataframe-operations");
header("Benchmark 02: DataFrame Operations");

// ── Helpers ───────────────────────────────────────────────

function makeData(n: number) {
  const names: string[] = [];
  const ages: number[] = [];
  const salaries: number[] = [];
  const departments: string[] = [];
  const scores: number[] = [];
  const depts = ["Engineering", "Sales", "HR", "Marketing", "Finance"];
  for (let i = 0; i < n; i++) {
    names.push(`Person_${i}`);
    ages.push(20 + (i % 50));
    salaries.push(30000 + ((i * 17) % 90000));
    departments.push(depts[i % depts.length]);
    scores.push(50 + ((i * 7) % 50));
  }
  return {
    name: names,
    age: ages,
    salary: salaries,
    department: departments,
    score: scores,
  };
}

// ── Creation ──────────────────────────────────────────────

run(suite, "DataFrame creation", "100 rows", () => {
  new DataFrame(makeData(100));
});

run(suite, "DataFrame creation", "1K rows", () => {
  new DataFrame(makeData(1000));
});

run(suite, "DataFrame creation", "10K rows", () => {
  new DataFrame(makeData(10000));
});

run(
  suite,
  "DataFrame creation",
  "50K rows",
  () => {
    new DataFrame(makeData(50000));
  },
  { iterations: 10 }
);

// ── Column Selection ──────────────────────────────────────

const df1k = new DataFrame(makeData(1000));
const df10k = new DataFrame(makeData(10000));
const df50k = new DataFrame(makeData(50000));

run(suite, "select single column", "1K rows", () => {
  df1k.get("salary");
});

run(suite, "select single column", "10K rows", () => {
  df10k.get("salary");
});

run(suite, "select multiple columns", "10K rows", () => {
  df10k.select(["name", "salary", "department"]);
});

run(suite, "select multiple columns", "50K rows", () => {
  df50k.select(["name", "salary", "department"]);
});

// ── Filtering ─────────────────────────────────────────────

run(suite, "filter rows (salary > 60000)", "1K rows", () => {
  df1k.filter((row) => (row.salary as number) > 60000);
});

run(suite, "filter rows (salary > 60000)", "10K rows", () => {
  df10k.filter((row) => (row.salary as number) > 60000);
});

run(
  suite,
  "filter rows (salary > 60000)",
  "50K rows",
  () => {
    df50k.filter((row) => (row.salary as number) > 60000);
  },
  { iterations: 10 }
);

// ── Sorting ───────────────────────────────────────────────

run(suite, "sort by salary", "1K rows", () => {
  df1k.sort("salary");
});

run(suite, "sort by salary", "10K rows", () => {
  df10k.sort("salary");
});

run(
  suite,
  "sort by salary",
  "50K rows",
  () => {
    df50k.sort("salary");
  },
  { iterations: 10 }
);

// ── GroupBy + Aggregation ─────────────────────────────────

run(suite, "groupBy + agg (sum)", "1K rows", () => {
  df1k.groupBy("department").agg({ salary: "sum" });
});

run(suite, "groupBy + agg (sum)", "10K rows", () => {
  df10k.groupBy("department").agg({ salary: "sum" });
});

run(
  suite,
  "groupBy + agg (sum)",
  "50K rows",
  () => {
    df50k.groupBy("department").agg({ salary: "sum" });
  },
  { iterations: 10 }
);

run(suite, "groupBy + agg (mean)", "10K rows", () => {
  df10k.groupBy("department").agg({ salary: "mean", score: "mean" });
});

run(
  suite,
  "groupBy + agg (mean)",
  "50K rows",
  () => {
    df50k.groupBy("department").agg({ salary: "mean", score: "mean" });
  },
  { iterations: 10 }
);

// ── Head / Tail ───────────────────────────────────────────

run(suite, "head(10)", "50K rows", () => {
  df50k.head(10);
});

run(suite, "tail(10)", "50K rows", () => {
  df50k.tail(10);
});

footer(suite, "deepbox-dataframe-ops.json");
