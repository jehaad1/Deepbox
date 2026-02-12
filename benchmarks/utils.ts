/**
 * Shared benchmark utilities for the Deepbox benchmark suite.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

// ── Types ─────────────────────────────────────────────────

export interface BenchmarkResult {
  operation: string;
  size: string;
  iterations: number;
  mean_ms: number;
  std_ms: number;
  min_ms: number;
  max_ms: number;
  ops_per_sec: number;
}

export interface BenchmarkSuite {
  benchmark: string;
  platform: string;
  timestamp: string;
  system: { runtime: string; version: string };
  results: BenchmarkResult[];
}

// ── Core ──────────────────────────────────────────────────

export function createSuite(name: string): BenchmarkSuite {
  return {
    benchmark: name,
    platform: "deepbox",
    timestamp: new Date().toISOString(),
    system: {
      runtime: "Node.js / tsx",
      version: process?.versions?.node ?? "unknown",
    },
    results: [],
  };
}

/**
 * Benchmark a function: warm up, then time N iterations.
 * Returns { mean, std, min, max } in milliseconds.
 */
function bench(
  fn: () => void,
  iterations = 20,
  warmup = 5
): { mean: number; std: number; min: number; max: number } {
  for (let i = 0; i < warmup; i++) fn();

  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    fn();
    times.push(performance.now() - start);
  }

  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  const std = Math.sqrt(times.reduce((s, t) => s + (t - mean) ** 2, 0) / times.length);
  return { mean, std, min: Math.min(...times), max: Math.max(...times) };
}

/**
 * Run a benchmark and add the result to the suite. Prints a row to console.
 */
export function run(
  suite: BenchmarkSuite,
  operation: string,
  size: string,
  fn: () => void,
  opts?: { iterations?: number; warmup?: number }
): void {
  const { mean, std, min, max } = bench(fn, opts?.iterations ?? 20, opts?.warmup ?? 5);
  const ops = mean > 0 ? 1000 / mean : Infinity;

  suite.results.push({
    operation,
    size,
    iterations: opts?.iterations ?? 20,
    mean_ms: +mean.toFixed(4),
    std_ms: +std.toFixed(4),
    min_ms: +min.toFixed(4),
    max_ms: +max.toFixed(4),
    ops_per_sec: +ops.toFixed(2),
  });

  const op = operation.padEnd(44);
  const sz = size.padEnd(12);
  const m = `${mean.toFixed(3)} ms`.padStart(10);
  const s = `${std.toFixed(3)} ms`.padStart(10);
  const o = `${Math.round(ops)}`.padStart(10);
  console.log(`  ${op} ${sz} ${m}  ± ${s}  (${o} ops/s)`);
}

// ── Print helpers ─────────────────────────────────────────

export function header(title: string): void {
  console.log("=".repeat(100));
  console.log(`  ${title}`);
  console.log(
    `  Platform: Deepbox (TypeScript) | Runtime: ${process?.versions?.node ?? "unknown"}`
  );
  console.log("=".repeat(100));
  console.log(
    `  ${"Operation".padEnd(44)} ${"Size".padEnd(12)} ${"Mean".padStart(10)}     ${"Std Dev".padStart(10)}  ${"Throughput".padStart(12)}`
  );
  console.log("-".repeat(100));
}

export function footer(suite: BenchmarkSuite, outputFile: string): void {
  console.log("-".repeat(100));
  console.log(`  Total: ${suite.results.length} benchmarks`);
  console.log("=".repeat(100));

  const outPath = `benchmarks/results/${outputFile}`;
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(suite, null, 2));
  console.log(`  ✓ Saved → ${outPath}\n`);
}
