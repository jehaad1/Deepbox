/**
 * Benchmark 01 — DataFrame Operations
 * Deepbox vs Pandas
 */

import { DataFrame } from "deepbox/dataframe";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("dataframe");
header("Benchmark 01 — DataFrame Operations");

// ── Data generators ──────────────────────────────────────

function seededRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

function makeNumericDF(nRows: number, nCols: number, seed: number): DataFrame {
  const rand = seededRng(seed);
  const data: Record<string, number[]> = {};
  for (let c = 0; c < nCols; c++) {
    const col: number[] = [];
    for (let r = 0; r < nRows; r++) col.push(rand() * 100);
    data[`col${c}`] = col;
  }
  return new DataFrame(data);
}

function makeMixedDF(nRows: number, seed: number): DataFrame {
  const rand = seededRng(seed);
  const names: string[] = [];
  const ages: number[] = [];
  const scores: number[] = [];
  const categories: string[] = [];
  const values: number[] = [];
  const cats = ["A", "B", "C", "D", "E"];
  for (let i = 0; i < nRows; i++) {
    names.push(`person_${i}`);
    ages.push(Math.floor(rand() * 60) + 18);
    scores.push(rand() * 100);
    categories.push(cats[Math.floor(rand() * cats.length)]);
    values.push(rand() * 1000);
  }
  return new DataFrame({
    name: names,
    age: ages,
    score: scores,
    category: categories,
    value: values,
  });
}

const _df100 = makeMixedDF(100, 42);
const df1k = makeMixedDF(1000, 42);
const df10k = makeMixedDF(10000, 42);
const df50k = makeMixedDF(50000, 42);
const numDf100 = makeNumericDF(100, 5, 42);
const numDf1k = makeNumericDF(1000, 5, 42);
const numDf10k = makeNumericDF(10000, 5, 42);

// ── Creation ─────────────────────────────────────────────

run(suite, "create", "100 rows", () => makeMixedDF(100, 99));
run(suite, "create", "1K rows", () => makeMixedDF(1000, 99));
run(suite, "create", "10K rows", () => makeMixedDF(10000, 99));
run(suite, "create (numeric)", "100x5", () => makeNumericDF(100, 5, 99));
run(suite, "create (numeric)", "1Kx5", () => makeNumericDF(1000, 5, 99));
run(suite, "create (numeric)", "10Kx5", () => makeNumericDF(10000, 5, 99));

// ── Select ───────────────────────────────────────────────

run(suite, "select (1 col)", "1K rows", () => df1k.select(["score"]));
run(suite, "select (1 col)", "10K rows", () => df10k.select(["score"]));
run(suite, "select (3 cols)", "1K rows", () => df1k.select(["name", "age", "score"]));
run(suite, "select (3 cols)", "10K rows", () => df10k.select(["name", "age", "score"]));

// ── Filter ───────────────────────────────────────────────

run(suite, "filter (numeric >)", "1K rows", () => df1k.filter((r) => (r.age as number) > 40));
run(suite, "filter (numeric >)", "10K rows", () => df10k.filter((r) => (r.age as number) > 40));
run(suite, "filter (string ==)", "1K rows", () => df1k.filter((r) => r.category === "A"));
run(suite, "filter (string ==)", "10K rows", () => df10k.filter((r) => r.category === "A"));
run(suite, "filter (compound)", "1K rows", () =>
  df1k.filter((r) => (r.age as number) > 30 && (r.score as number) > 50)
);
run(suite, "filter (compound)", "10K rows", () =>
  df10k.filter((r) => (r.age as number) > 30 && (r.score as number) > 50)
);

// ── Sort ─────────────────────────────────────────────────

run(suite, "sort (single col)", "1K rows", () => df1k.sort("age"));
run(suite, "sort (single col)", "10K rows", () => df10k.sort("age"));
run(suite, "sort (descending)", "1K rows", () => df1k.sort("score", false));
run(suite, "sort (descending)", "10K rows", () => df10k.sort("score", false));

// ── GroupBy ──────────────────────────────────────────────

run(suite, "groupBy + sum", "1K rows", () => df1k.groupBy("category").sum());
run(suite, "groupBy + sum", "10K rows", () => df10k.groupBy("category").sum());
run(suite, "groupBy + mean", "1K rows", () => df1k.groupBy("category").mean());
run(suite, "groupBy + mean", "10K rows", () => df10k.groupBy("category").mean());
run(suite, "groupBy + count", "1K rows", () => df1k.groupBy("category").count());
run(suite, "groupBy + count", "10K rows", () => df10k.groupBy("category").count());
run(suite, "groupBy + min", "1K rows", () => df1k.groupBy("category").min());
run(suite, "groupBy + min", "10K rows", () => df10k.groupBy("category").min());
run(suite, "groupBy + max", "1K rows", () => df1k.groupBy("category").max());
run(suite, "groupBy + max", "10K rows", () => df10k.groupBy("category").max());

// ── Head / Tail ──────────────────────────────────────────

run(suite, "head(10)", "10K rows", () => df10k.head(10));
run(suite, "head(10)", "50K rows", () => df50k.head(10));
run(suite, "tail(10)", "10K rows", () => df10k.tail(10));
run(suite, "tail(10)", "50K rows", () => df50k.tail(10));

// ── Loc / Iloc ───────────────────────────────────────────

run(suite, "iloc(0)", "1K rows", () => df1k.iloc(0));
run(suite, "iloc(500)", "1K rows", () => df1k.iloc(500));
run(suite, "iloc(0)", "10K rows", () => df10k.iloc(0));
run(suite, "iloc(5000)", "10K rows", () => df10k.iloc(5000));

// ── Join ─────────────────────────────────────────────────

const leftDf = new DataFrame({
  key: Array.from({ length: 500 }, (_, i) => i % 50),
  val_l: Array.from({ length: 500 }, (_, i) => i),
});
const rightDf = new DataFrame({
  key: Array.from({ length: 50 }, (_, i) => i),
  val_r: Array.from({ length: 50 }, (_, i) => i * 10),
});
const leftDf2k = new DataFrame({
  key: Array.from({ length: 2000 }, (_, i) => i % 100),
  val_l: Array.from({ length: 2000 }, (_, i) => i),
});
const rightDf100 = new DataFrame({
  key: Array.from({ length: 100 }, (_, i) => i),
  val_r: Array.from({ length: 100 }, (_, i) => i * 10),
});

run(suite, "join (inner)", "500×50", () => leftDf.join(rightDf, "key", "inner"));
run(suite, "join (left)", "500×50", () => leftDf.join(rightDf, "key", "left"));
run(suite, "join (inner)", "2K×100", () => leftDf2k.join(rightDf100, "key", "inner"));
run(suite, "join (left)", "2K×100", () => leftDf2k.join(rightDf100, "key", "left"));

// ── Concat ───────────────────────────────────────────────

run(suite, "concat (axis=0)", "2×1K", () => df1k.concat(df1k, 0));
run(suite, "concat (axis=0)", "2×10K", () => df10k.concat(df10k, 0));

// ── FillNa / DropNa ─────────────────────────────────────

const dfWithNa1k = new DataFrame({
  a: Array.from({ length: 1000 }, (_, i) => (i % 5 === 0 ? null : i)),
  b: Array.from({ length: 1000 }, (_, i) => (i % 3 === 0 ? null : i * 2)),
});
const dfWithNa10k = new DataFrame({
  a: Array.from({ length: 10000 }, (_, i) => (i % 5 === 0 ? null : i)),
  b: Array.from({ length: 10000 }, (_, i) => (i % 3 === 0 ? null : i * 2)),
});

run(suite, "fillna(0)", "1K rows", () => dfWithNa1k.fillna(0));
run(suite, "fillna(0)", "10K rows", () => dfWithNa10k.fillna(0));
run(suite, "dropna", "1K rows", () => dfWithNa1k.dropna());
run(suite, "dropna", "10K rows", () => dfWithNa10k.dropna());

// ── Describe ─────────────────────────────────────────────

run(suite, "describe", "100x5", () => numDf100.describe());
run(suite, "describe", "1Kx5", () => numDf1k.describe());
run(suite, "describe", "10Kx5", () => numDf10k.describe());

// ── Correlation ──────────────────────────────────────────

run(suite, "corr", "100x5", () => numDf100.corr());
run(suite, "corr", "1Kx5", () => numDf1k.corr());

// ── Drop ─────────────────────────────────────────────────

run(suite, "drop (1 col)", "1K rows", () => df1k.drop(["value"]));
run(suite, "drop (2 cols)", "10K rows", () => df10k.drop(["value", "name"]));

footer(suite, "deepbox-dataframe.json");
