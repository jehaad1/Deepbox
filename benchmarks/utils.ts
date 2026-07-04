/**
 * Shared benchmark utilities for the Deepbox benchmark suite.
 *
 * Schema v2 adds:
 * - adaptive batching for very fast operations
 * - median/p95 statistics for more stable winner selection
 * - environment + methodology metadata
 * - comparable/deepbox-only case tracking
 */

import { mkdirSync, writeFileSync } from "node:fs";
import os from "node:os";
import { dirname } from "node:path";

export const BENCHMARK_SCHEMA_VERSION = 2;

export interface BenchmarkResult {
  id: string;
  operation: string;
  size: string;
  comparable: boolean;
  scope: "match" | "local";
  tags: string[];
  warmup: number;
  samples: number;
  batch_size: number;
  total_invocations: number;
  mean_ms: number;
  median_ms: number;
  std_ms: number;
  min_ms: number;
  p95_ms: number;
  max_ms: number;
  ops_per_sec: number;
  relative_std_pct: number;
}

export interface BenchmarkSuite {
  schema_version: number;
  benchmark: string;
  platform: string;
  timestamp: string;
  system: {
    runtime: string;
    version: string;
    os: string;
    release: string;
    arch: string;
    cpu: string;
  };
  methodology: {
    timer: string;
    compare_metric: "median_ms";
    default_warmup: number;
    default_min_samples: number;
    default_max_samples: number;
    target_sample_ms: number;
    target_total_ms: number;
    adaptive_batching: boolean;
    gc_between_samples: boolean;
  };
  results: BenchmarkResult[];
}

export interface RunOptions {
  iterations?: number;
  warmup?: number;
  minSamples?: number;
  maxSamples?: number;
  targetSampleMs?: number;
  targetTotalMs?: number;
  comparable?: boolean;
  tags?: string[];
  id?: string;
}

const DEFAULT_WARMUP = 5;
const DEFAULT_MIN_SAMPLES = 20;
const DEFAULT_MAX_SAMPLES = 60;
const DEFAULT_TARGET_SAMPLE_MS = 5;
const DEFAULT_TARGET_TOTAL_MS = 250;
const MAX_BATCH_SIZE = 10_000;

let blackhole = 0;

function consume(value: unknown): void {
  if (typeof value === "number" && Number.isFinite(value)) {
    blackhole ^= Math.trunc(value * 1_000_000);
    return;
  }
  if (typeof value === "string") {
    blackhole ^= value.length;
    return;
  }
  if (typeof value === "boolean") {
    blackhole ^= value ? 1 : 0;
    return;
  }
  if (typeof value === "bigint") {
    blackhole ^= Number(value & 0xffffn);
    return;
  }
  if (Array.isArray(value)) {
    blackhole ^= value.length;
    return;
  }
  // Typed arrays / DataViews are ArrayBuffer views, not plain objects. Consume
  // them by length like Array — `Object.keys()` on a large typed array would
  // materialize one index-key string per element (O(n) with heavy allocation),
  // which would dwarf the operation being measured for functions that return a
  // raw typed array (e.g. Generator.randomArray).
  if (ArrayBuffer.isView(value)) {
    blackhole ^= (value as ArrayBufferView & { length?: number }).length ?? 0;
    return;
  }
  if (value && typeof value === "object") {
    blackhole ^= Object.keys(value).length;
  }
}

function nowNs(): bigint {
  const nodeProcess = globalThis as typeof globalThis & {
    process?: { hrtime?: { bigint?: () => bigint } };
  };
  if (typeof nodeProcess.process?.hrtime?.bigint === "function") {
    return nodeProcess.process.hrtime.bigint();
  }
  return BigInt(Math.floor(performance.now() * 1_000_000));
}

function toMs(durationNs: bigint): number {
  return Number(durationNs) / 1_000_000;
}

function round(value: number, digits = 4): number {
  return Number(value.toFixed(digits));
}

function percentile(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0] ?? 0;
  const pos = (sorted.length - 1) * q;
  const lower = Math.floor(pos);
  const upper = Math.ceil(pos);
  const lowerValue = sorted[lower] ?? 0;
  const upperValue = sorted[upper] ?? lowerValue;
  if (lower === upper) return lowerValue;
  return lowerValue + (upperValue - lowerValue) * (pos - lower);
}

function computeStats(samples: readonly number[]) {
  const sorted = [...samples].sort((a, b) => a - b);
  const mean = samples.reduce((acc, value) => acc + value, 0) / samples.length;
  const variance = samples.reduce((acc, value) => acc + (value - mean) ** 2, 0) / samples.length;
  const std = Math.sqrt(variance);
  const median = percentile(sorted, 0.5);
  const p95 = percentile(sorted, 0.95);
  const min = sorted[0] ?? 0;
  const max = sorted[sorted.length - 1] ?? 0;
  return { mean, median, std, min, p95, max };
}

function scopeOf(comparable: boolean): "match" | "local" {
  return comparable ? "match" : "local";
}

function logResult(
  operation: string,
  size: string,
  stats: ReturnType<typeof computeStats>,
  relativeStdPct: number,
  sampleCount: number,
  batchSize: number,
  comparable: boolean
): void {
  const op = operation.padEnd(34);
  const sz = size.padEnd(14);
  const median = `${stats.median.toFixed(3)} ms`.padStart(12);
  const p95 = `${stats.p95.toFixed(3)} ms`.padStart(12);
  const cv = `${relativeStdPct.toFixed(1)}%`.padStart(8);
  const sampleInfo = `${sampleCount}x${batchSize}`.padStart(9);
  console.log(`  ${op} ${sz} ${median} ${p95} ${cv} ${sampleInfo}  ${scopeOf(comparable)}`);
}

function pushResult(
  suite: BenchmarkSuite,
  operation: string,
  size: string,
  stats: ReturnType<typeof computeStats>,
  sampleCount: number,
  batchSize: number,
  warmup: number,
  opts: RunOptions
): number {
  const comparable = opts.comparable ?? true;
  const opsPerSec = stats.median > 0 ? 1000 / stats.median : Number.POSITIVE_INFINITY;
  const relativeStdPct = stats.mean > 0 ? (stats.std / stats.mean) * 100 : 0;

  suite.results.push({
    id: opts.id ?? makeId(suite.benchmark, operation, size),
    operation,
    size,
    comparable,
    scope: scopeOf(comparable),
    tags: opts.tags ?? [],
    warmup,
    samples: sampleCount,
    batch_size: batchSize,
    total_invocations: batchSize * (warmup + sampleCount),
    mean_ms: round(stats.mean),
    median_ms: round(stats.median),
    std_ms: round(stats.std),
    min_ms: round(stats.min),
    p95_ms: round(stats.p95),
    max_ms: round(stats.max),
    ops_per_sec: round(opsPerSec, 2),
    relative_std_pct: round(relativeStdPct, 2),
  });

  logResult(operation, size, stats, relativeStdPct, sampleCount, batchSize, comparable);
  return relativeStdPct;
}

function validateSuite(suite: BenchmarkSuite): void {
  const idSet = new Set<string>();
  const caseKeySet = new Set<string>();
  for (const result of suite.results) {
    if (idSet.has(result.id)) {
      throw new Error(`Duplicate benchmark id detected: ${result.id}`);
    }
    idSet.add(result.id);

    const caseKey = `${result.operation}::${result.size}`;
    if (caseKeySet.has(caseKey)) {
      throw new Error(
        `Duplicate benchmark operation/size detected in suite "${suite.benchmark}": ${caseKey}`
      );
    }
    caseKeySet.add(caseKey);
  }
}

function maybeGc(): void {
  const gc = (globalThis as { gc?: () => void }).gc;
  if (typeof gc === "function") {
    gc();
  }
}

function makeId(benchmark: string, operation: string, size: string): string {
  const slug = `${benchmark}-${operation}-${size}`
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug || `${benchmark}-case`;
}

function formatCpu(): string {
  const cpus = os.cpus();
  return cpus[0]?.model ?? "unknown";
}

function measureCase(
  fn: () => unknown,
  opts: Required<Pick<RunOptions, "warmup" | "targetSampleMs" | "targetTotalMs">> &
    Pick<RunOptions, "iterations" | "minSamples" | "maxSamples">
) {
  for (let i = 0; i < opts.warmup; i++) {
    consume(fn());
  }

  maybeGc();

  const calibrationStart = nowNs();
  consume(fn());
  const calibrationMs = Math.max(toMs(nowNs() - calibrationStart), 0.0001);

  const batchSize = Math.max(
    1,
    Math.min(MAX_BATCH_SIZE, Math.ceil(opts.targetSampleMs / calibrationMs))
  );

  const targetSamples = opts.iterations;
  const minSamples = targetSamples ?? opts.minSamples ?? DEFAULT_MIN_SAMPLES;
  const maxSamples = targetSamples ?? opts.maxSamples ?? DEFAULT_MAX_SAMPLES;
  const fixedSamples = targetSamples !== undefined;

  const samples: number[] = [];
  let totalElapsedMs = 0;

  while (samples.length < maxSamples) {
    maybeGc();
    const start = nowNs();
    for (let i = 0; i < batchSize; i++) {
      consume(fn());
    }
    const elapsedMs = toMs(nowNs() - start) / batchSize;
    samples.push(elapsedMs);
    totalElapsedMs += elapsedMs * batchSize;

    if (fixedSamples && samples.length >= targetSamples) break;
    if (!fixedSamples && samples.length >= minSamples && totalElapsedMs >= opts.targetTotalMs)
      break;
  }

  return {
    stats: computeStats(samples),
    samples,
    batchSize,
  };
}

export function createSuite(name: string): BenchmarkSuite {
  return {
    schema_version: BENCHMARK_SCHEMA_VERSION,
    benchmark: name,
    platform: "deepbox",
    timestamp: new Date().toISOString(),
    system: {
      runtime: "Node.js / tsx",
      version: process?.versions?.node ?? "unknown",
      os: os.platform(),
      release: os.release(),
      arch: os.arch(),
      cpu: formatCpu(),
    },
    methodology: {
      timer: "process.hrtime.bigint",
      compare_metric: "median_ms",
      default_warmup: DEFAULT_WARMUP,
      default_min_samples: DEFAULT_MIN_SAMPLES,
      default_max_samples: DEFAULT_MAX_SAMPLES,
      target_sample_ms: DEFAULT_TARGET_SAMPLE_MS,
      target_total_ms: DEFAULT_TARGET_TOTAL_MS,
      adaptive_batching: true,
      gc_between_samples: typeof (globalThis as { gc?: () => void }).gc === "function",
    },
    results: [],
  };
}

export function run(
  suite: BenchmarkSuite,
  operation: string,
  size: string,
  fn: () => unknown,
  opts: RunOptions = {}
): void {
  const warmup = opts.warmup ?? DEFAULT_WARMUP;
  const { stats, samples, batchSize } = measureCase(fn, {
    iterations: opts.iterations,
    warmup,
    minSamples: opts.minSamples,
    maxSamples: opts.maxSamples,
    targetSampleMs: opts.targetSampleMs ?? suite.methodology.target_sample_ms,
    targetTotalMs: opts.targetTotalMs ?? suite.methodology.target_total_ms,
  });
  pushResult(suite, operation, size, stats, samples.length, batchSize, warmup, opts);
}

export async function runAsync(
  suite: BenchmarkSuite,
  operation: string,
  size: string,
  fn: () => Promise<unknown>,
  opts: RunOptions = {}
): Promise<void> {
  const wrapper = async () => fn();
  const warmup = opts.warmup ?? DEFAULT_WARMUP;
  const targetSampleMs = opts.targetSampleMs ?? suite.methodology.target_sample_ms;
  const targetTotalMs = opts.targetTotalMs ?? suite.methodology.target_total_ms;
  const targetSamples = opts.iterations;
  const minSamples = targetSamples ?? opts.minSamples ?? DEFAULT_MIN_SAMPLES;
  const maxSamples = targetSamples ?? opts.maxSamples ?? DEFAULT_MAX_SAMPLES;
  const fixedSamples = targetSamples !== undefined;

  for (let i = 0; i < warmup; i++) {
    consume(await wrapper());
  }

  maybeGc();

  const calibrationStart = nowNs();
  consume(await wrapper());
  const calibrationMs = Math.max(toMs(nowNs() - calibrationStart), 0.0001);
  const batchSize = Math.max(1, Math.min(1000, Math.ceil(targetSampleMs / calibrationMs)));

  const samples: number[] = [];
  let totalElapsedMs = 0;

  while (samples.length < maxSamples) {
    maybeGc();
    const start = nowNs();
    for (let i = 0; i < batchSize; i++) {
      consume(await wrapper());
    }
    const elapsedMs = toMs(nowNs() - start) / batchSize;
    samples.push(elapsedMs);
    totalElapsedMs += elapsedMs * batchSize;

    if (fixedSamples && samples.length >= targetSamples) break;
    if (!fixedSamples && samples.length >= minSamples && totalElapsedMs >= targetTotalMs) break;
  }

  const stats = computeStats(samples);
  pushResult(suite, operation, size, stats, samples.length, batchSize, warmup, opts);
}

export function header(title: string): void {
  console.log("=".repeat(116));
  console.log(`  ${title}`);
  console.log(
    `  Platform: Deepbox (TypeScript) | Runtime: Node ${process?.versions?.node ?? "unknown"} | Compare metric: median_ms`
  );
  console.log(
    `  Sampling: adaptive batching, ${DEFAULT_MIN_SAMPLES}-${DEFAULT_MAX_SAMPLES} samples, target ${DEFAULT_TARGET_TOTAL_MS} ms/case`
  );
  console.log("=".repeat(116));
  console.log(
    `  ${"Operation".padEnd(34)} ${"Size".padEnd(14)} ${"Median".padStart(12)} ${"P95".padStart(12)} ${"CV".padStart(8)} ${"Samples".padStart(9)}  Scope`
  );
  console.log("-".repeat(116));
}

export function footer(suite: BenchmarkSuite, outputFile: string): void {
  validateSuite(suite);
  console.log("-".repeat(116));
  const comparableCount = suite.results.filter((result) => result.comparable).length;
  const localCount = suite.results.length - comparableCount;
  console.log(
    `  Total: ${suite.results.length} benchmarks (${comparableCount} comparable, ${localCount} local)`
  );
  console.log(`  Blackhole: ${blackhole}`);
  console.log("=".repeat(116));

  const outPath = `benchmarks/results/${outputFile}`;
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(suite, null, 2));
  console.log(`  ✓ Saved → ${outPath}\n`);
}
