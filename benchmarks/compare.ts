/**
 * Benchmark Comparison Tool
 *
 * Reads Deepbox and Python JSON results from benchmarks/results/
 * and generates RESULTS.md with side-by-side comparison tables.
 *
 * Usage: npm run bench:compare
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";

interface Result {
  operation: string;
  size: string;
  mean_ms?: number;
  median_ms?: number;
  comparable?: boolean;
  tags?: string[];
}

/**
 * Timer-floor threshold, in milliseconds.
 *
 * Deepbox's lazy-view operations (transpose/reshape/head/tail/iloc as strided
 * views, spec-build plot appends) complete in ~0.10–0.50 µs regardless of input
 * size, because they only rewrite shape/stride metadata and defer the actual
 * work. Below ~1 µs we are measuring the harness/timer floor and V8's ability
 * to elide deferred work, not sustained throughput. A Python library that
 * *materializes* the same result (`.T.copy()`, a fresh DataFrame) inevitably
 * looks "slower" here, producing eye-popping but meaningless speedups.
 *
 * The **realized-work** win rate excludes every comparable case whose Deepbox
 * median falls below this floor (or that is explicitly tagged as a lazy view /
 * spec build). It reflects throughput on operations that actually move data.
 */
const REALIZED_FLOOR_MS = 0.001;

/** Tags that mark a case as architectural deferral rather than realized work. */
const DEFERRED_TAGS = new Set(["lazy-view", "spec-build"]);

interface Suite {
  benchmark: string;
  platform: string;
  results: Result[];
}

interface Row {
  operation: string;
  size: string;
  deepbox_ms: number;
  python_ms: number;
  speedup: number;
  winner: "Deepbox" | "lib";
  /**
   * True when this row measures realized work (moves/materializes data) and is
   * therefore eligible for the realized-work win rate. False for sub-microsecond
   * timer-floor lazy views / spec builds.
   */
  realized: boolean;
}

const BENCHMARKS = [
  {
    name: "DataFrame Operations",
    deepbox: "deepbox-dataframe.json",
    python: "pandas-dataframe.json",
    lib: "Pandas",
  },
  {
    name: "Dataset Loading",
    deepbox: "deepbox-datasets.json",
    python: "sklearn-datasets.json",
    lib: "scikit-learn",
  },
  {
    name: "Linear Algebra",
    deepbox: "deepbox-linalg.json",
    python: "numpy-linalg.json",
    lib: "NumPy/SciPy",
  },
  {
    name: "Metrics",
    deepbox: "deepbox-metrics.json",
    python: "sklearn-metrics.json",
    lib: "scikit-learn",
  },
  {
    name: "ML Training",
    deepbox: "deepbox-ml.json",
    python: "sklearn-ml.json",
    lib: "scikit-learn",
  },
  {
    name: "NDArray Operations",
    deepbox: "deepbox-ndarray.json",
    python: "numpy-ndarray.json",
    lib: "NumPy",
  },
  {
    name: "Neural Networks",
    deepbox: "deepbox-nn.json",
    python: "pytorch-nn.json",
    lib: "PyTorch",
  },
  {
    name: "Optimizers",
    deepbox: "deepbox-optim.json",
    python: "pytorch-optim.json",
    lib: "PyTorch",
  },
  {
    name: "Plotting",
    deepbox: "deepbox-plot.json",
    python: "matplotlib-plot.json",
    lib: "Matplotlib",
  },
  {
    name: "Preprocessing",
    deepbox: "deepbox-preprocess.json",
    python: "sklearn-preprocess.json",
    lib: "scikit-learn",
  },
  {
    name: "Random Generation",
    deepbox: "deepbox-random.json",
    python: "numpy-random.json",
    lib: "NumPy",
  },
  {
    name: "Statistical Analysis",
    deepbox: "deepbox-stats.json",
    python: "scipy-stats.json",
    lib: "SciPy",
  },
];

const PYTHON_SCRIPT_HINTS: Record<string, string> = {
  "pandas-dataframe.json": "benchmarks/python/01_dataframe.py",
  "sklearn-datasets.json": "benchmarks/python/02_datasets.py",
  "numpy-linalg.json": "benchmarks/python/03_linalg.py",
  "sklearn-metrics.json": "benchmarks/python/04_metrics.py",
  "sklearn-ml.json": "benchmarks/python/05_ml.py",
  "numpy-ndarray.json": "benchmarks/python/06_ndarray.py",
  "pytorch-nn.json": "benchmarks/python/07_nn.py",
  "pytorch-optim.json": "benchmarks/python/08_optim.py",
  "matplotlib-plot.json": "benchmarks/python/09_plot.py",
  "sklearn-preprocess.json": "benchmarks/python/10_preprocess.py",
  "numpy-random.json": "benchmarks/python/11_random.py",
  "scipy-stats.json": "benchmarks/python/12_stats.py",
};

function load(file: string): Suite | null {
  const path = `benchmarks/results/${file}`;
  if (!existsSync(path)) return null;
  return JSON.parse(readFileSync(path, "utf-8"));
}

function metric(result: Result): number {
  return result.median_ms ?? result.mean_ms ?? 0;
}

function compare(db: Result[], py: Result[]): Row[] {
  const dbComparable = db.filter((r) => r.comparable ?? true);
  const pyComparable = py.filter((r) => r.comparable ?? true);
  const pyMap = new Map(pyComparable.map((r) => [`${r.operation}|${r.size}`, r]));
  const rows: Row[] = [];
  for (const d of dbComparable) {
    const p = pyMap.get(`${d.operation}|${d.size}`);
    if (!p) continue;
    const deepboxMs = metric(d);
    const pythonMs = metric(p);
    if (deepboxMs <= 0 || pythonMs <= 0) continue;
    const speedup = pythonMs / deepboxMs;
    const winner: Row["winner"] = speedup >= 1 ? "Deepbox" : "lib";
    const tagged = (d.tags ?? []).some((tag) => DEFERRED_TAGS.has(tag));
    const realized = deepboxMs >= REALIZED_FLOOR_MS && !tagged;
    rows.push({
      operation: d.operation,
      size: d.size,
      deepbox_ms: deepboxMs,
      python_ms: pythonMs,
      speedup,
      winner,
      realized,
    });
  }
  return rows;
}

function fmtMs(ms: number): string {
  if (ms < 0.001) return `${(ms * 1000).toFixed(2)} µs`;
  if (ms < 1) return `${ms.toFixed(3)} ms`;
  if (ms < 1000) return `${ms.toFixed(1)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function fmtSpeed(s: number): string {
  return s >= 1 ? `${s.toFixed(1)}x faster` : `${(1 / s).toFixed(1)}x slower`;
}

const ORANGE = "\u{1F7E0}";
const GREEN = "\u{1F7E2}";

// ── Run ───────────────────────────────────────────────────

let totalDs = 0,
  totalLib = 0;
let realizedDs = 0,
  realizedLib = 0;
let totalLocalOnly = 0;
const sections: { name: string; lib: string; rows: Row[] }[] = [];

console.log(`\n${"=".repeat(100)}`);
console.log("  DEEPBOX vs PYTHON PACKAGES — BENCHMARK COMPARISON");
console.log(`${"=".repeat(100)}\n`);

for (const b of BENCHMARKS) {
  const db = load(b.deepbox);
  const py = load(b.python);
  if (!db) {
    console.log(`  ⚠ Missing: ${b.deepbox}`);
    sections.push({ name: b.name, lib: b.lib, rows: [] });
    continue;
  }
  if (!py) {
    const hint = PYTHON_SCRIPT_HINTS[b.python] ?? "benchmarks/python/<script>.py";
    console.log(`  ⚠ Missing: ${b.python} — run: python3 ${hint}`);
    sections.push({ name: b.name, lib: b.lib, rows: [] });
    continue;
  }

  const rows = compare(db.results, py.results);
  totalLocalOnly += db.results.filter((result) => !(result.comparable ?? true)).length;
  sections.push({ name: b.name, lib: b.lib, rows });

  const dw = rows.filter((r) => r.winner === "Deepbox").length;
  const lw = rows.filter((r) => r.winner === "lib").length;
  totalDs += dw;
  totalLib += lw;
  realizedDs += rows.filter((r) => r.realized && r.winner === "Deepbox").length;
  realizedLib += rows.filter((r) => r.realized && r.winner === "lib").length;

  console.log(`  ${b.name}: Deepbox ${dw} | ${b.lib} ${lw}`);
}

const total = totalDs + totalLib;
const pct = (n: number) => (total > 0 ? `${((n / total) * 100).toFixed(1)}%` : "—");
const realizedTotal = realizedDs + realizedLib;
const realizedPct = (n: number) =>
  realizedTotal > 0 ? `${((n / realizedTotal) * 100).toFixed(1)}%` : "—";
const floorExcluded = total - realizedTotal;

console.log(`\n${"-".repeat(60)}`);
console.log(`  OVERALL: ${total} comparable cases`);
console.log(`  ${ORANGE} Deepbox wins:         ${totalDs} (${pct(totalDs)})`);
console.log(`  ${GREEN} Python packages win:  ${totalLib} (${pct(totalLib)})`);
console.log(
  `\n  REALIZED-WORK: ${realizedTotal} cases (excludes ${floorExcluded} sub-µs lazy-view / spec-build)`
);
console.log(`  ${ORANGE} Deepbox wins:         ${realizedDs} (${realizedPct(realizedDs)})`);
console.log(`  ${GREEN} Python packages win:  ${realizedLib} (${realizedPct(realizedLib)})`);
console.log(`\n  Local-only Deepbox cases excluded entirely: ${totalLocalOnly}`);
console.log(`${"=".repeat(100)}\n`);

// ── Generate RESULTS.md ───────────────────────────────────

let md = `# Benchmark Results — Deepbox vs Python Packages\n\n`;
md += `> Auto-generated by \`benchmarks/compare.ts\` on ${new Date().toISOString().split("T")[0]}\n\n`;

md += `## Summary\n\n`;
md += `Two win rates are reported. The **overall** rate counts every comparable head-to-head case. The **realized-work** rate excludes ${floorExcluded} sub-microsecond lazy-view / spec-build cases (Deepbox median < ${(REALIZED_FLOOR_MS * 1000).toFixed(0)} µs, or tagged \`lazy-view\`/\`spec-build\`) where Deepbox only rewrites shape/stride metadata and defers the actual work — see the [note on lazy-view operations](#a-note-on-lazy-view-operations) below.\n\n`;
md += `### Overall (all comparable cases)\n\n`;
md += `| | Count | % |\n`;
md += `| --- | ---: | ---: |\n`;
md += `| ${ORANGE} Deepbox wins | ${totalDs} | ${pct(totalDs)} |\n`;
md += `| ${GREEN} Python packages win | ${totalLib} | ${pct(totalLib)} |\n`;
md += `| **Total** | **${total}** | |\n\n`;
md += `### Realized-work (materializing operations only)\n\n`;
md += `| | Count | % |\n`;
md += `| --- | ---: | ---: |\n`;
md += `| ${ORANGE} Deepbox wins | ${realizedDs} | ${realizedPct(realizedDs)} |\n`;
md += `| ${GREEN} Python packages win | ${realizedLib} | ${realizedPct(realizedLib)} |\n`;
md += `| **Total** | **${realizedTotal}** | |\n\n`;
md += `Sub-microsecond lazy-view / spec-build cases excluded from the realized-work rate: **${floorExcluded}**\n\n`;
md += `Local-only Deepbox benchmark cases recorded in JSON but excluded from both rates: **${totalLocalOnly}**\n\n`;

for (const sec of sections) {
  md += `---\n\n## ${sec.name} (vs ${sec.lib})\n\n`;
  if (sec.rows.length === 0) {
    md += `> No results yet. Run both Deepbox and ${sec.lib} benchmarks first.\n\n`;
    continue;
  }

  const dw = sec.rows.filter((r) => r.winner === "Deepbox").length;
  const lw = sec.rows.filter((r) => r.winner === "lib").length;
  const rdw = sec.rows.filter((r) => r.realized && r.winner === "Deepbox").length;
  const rlw = sec.rows.filter((r) => r.realized && r.winner === "lib").length;
  const floorRows = sec.rows.filter((r) => !r.realized).length;
  md += `**Overall: Deepbox ${dw} — ${sec.lib} ${lw}**`;
  if (floorRows > 0) {
    md += ` · **Realized-work: Deepbox ${rdw} — ${sec.lib} ${rlw}** (${floorRows} sub-µs lazy-view/spec-build case${floorRows === 1 ? "" : "s"} marked † and excluded from realized-work)`;
  }
  md += `\n\n`;
  md += `| Operation | Size | Deepbox | ${sec.lib} | Speedup | Winner |\n`;
  md += `| --- | --- | ---: | ---: | --- | --- |\n`;
  for (const r of sec.rows) {
    const icon = r.winner === "Deepbox" ? ORANGE : GREEN;
    const label = r.winner === "Deepbox" ? "Deepbox" : sec.lib;
    const marker = r.realized ? "" : " †";
    md += `| ${r.operation}${marker} | ${r.size} | ${fmtMs(r.deepbox_ms)} | ${fmtMs(r.python_ms)} | ${fmtSpeed(r.speedup)} | ${icon} ${label} |\n`;
  }
  md += `\n`;
}

md += `---\n\n## A note on lazy-view operations\n\n`;
md += `Rows marked **†** are **lazy-view or spec-build** operations whose Deepbox median sits below the ${(REALIZED_FLOOR_MS * 1000).toFixed(0)} µs timer floor. Deepbox implements \`transpose\`, \`reshape\`, \`head\`/\`tail\`/\`iloc\`, slicing, and figure assembly as **deferred, size-independent metadata rewrites** — a lazy strided view or an in-memory spec — with no data movement until a materializing consumer (\`.copy()\`, \`show()\`, iteration) forces it. The Python side of these cases usually *materializes* a fresh array/DataFrame/Figure, so the reported speedup measures **architectural deferral, not throughput**: it stays roughly constant no matter how large the input is, which is the tell-tale signature of a floor comparison rather than a compute win.\n\n`;
md += `To keep the comparison honest:\n\n`;
md += `- \`transpose\`, \`flatten\`, and \`slice\` in the NDArray suite now wrap the Deepbox result in \`copy(...)\` so **both** sides materialize a contiguous buffer (matching NumPy's \`.T.copy()\`/\`.flatten()\`/\`[…].copy()\`). These are genuine realized-work comparisons.\n`;
md += `- Plot spec-build cases (\`scatter\`, \`bar\`, \`pie\`, …) that only append to a figure spec are recorded as **local (non-comparable)** and excluded from both rates; the \`show (SVG)\`/\`show (PNG)\`/\`saveFig (PDF)\` cases render on both sides and remain the comparable plotting benchmarks.\n`;
md += `- The **realized-work** rate additionally drops any remaining sub-µs lazy view so the headline number reflects operations that actually move data.\n\n`;
md += `## Notes\n\n`;
md += `- **Speedup > 1x** = Deepbox is faster\n`;
md += `- **Speedup < 1x** = Python package is faster\n`;
md += `- Two win rates are reported: **overall** (all comparable cases) and **realized-work** (excludes sub-µs lazy-view/spec-build cases marked †)\n`;
md += `- Winner tables use **median_ms** for stability; local-only cases are excluded from both totals\n`;
md += `- Harness is symmetric: identical warmup, adaptive batching, sample counts, and median selection on both sides\n`;
md += `- NumPy/SciPy use C/Fortran BLAS backends; Deepbox is pure TypeScript on V8\n`;
md += `- PyTorch uses C++ ATen backend; CPU-only measurements\n`;
md += `- scikit-learn uses Cython/C extensions for core algorithms\n`;
md += `- Results are hardware-dependent — always compare on the same machine\n`;

writeFileSync("benchmarks/RESULTS.md", md);
console.log(`\n  ✓ Written → benchmarks/RESULTS.md`);

// ── Auto-update root README.md performance section ──────

const readmePath = "README.md";
if (existsSync(readmePath)) {
  let readme = readFileSync(readmePath, "utf-8");

  const perfStart = readme.indexOf("## Performance");
  const nextH2 = readme.indexOf("\n## ", perfStart + 15);
  if (perfStart !== -1 && nextH2 !== -1) {
    const TABLE_MAP: Record<string, { category: string; against: string }> = {
      "DataFrame Operations": {
        category: "DataFrames",
        against: "Pandas (C / Cython)",
      },
      "Dataset Loading": { category: "Datasets", against: "scikit-learn" },
      "Linear Algebra": {
        category: "Linear Algebra",
        against: "NumPy + SciPy (LAPACK)",
      },
      Metrics: { category: "Metrics", against: "scikit-learn (C / Cython)" },
      "ML Training": {
        category: "ML Training",
        against: "scikit-learn (C / Cython)",
      },
      "NDArray Operations": {
        category: "NDArray Ops",
        against: "NumPy (C / BLAS)",
      },
      "Neural Networks": {
        category: "Neural Networks",
        against: "PyTorch (C++ ATen)",
      },
      Optimizers: { category: "Optimizers", against: "PyTorch (C++ ATen)" },
      Plotting: { category: "Plotting", against: "Matplotlib (C / Agg)" },
      Preprocessing: {
        category: "Preprocessing",
        against: "scikit-learn (C / Cython)",
      },
      "Random Generation": { category: "Random", against: "NumPy (C)" },
      "Statistical Analysis": {
        category: "Statistics",
        against: "SciPy (C / Fortran)",
      },
    };

    const bestPerCat = new Map<string, { op: string; size: string; speed: number; cat: string }>();
    let tableRows = "";
    for (const sec of sections) {
      const info = TABLE_MAP[sec.name];
      if (!info || sec.rows.length === 0) continue;
      const dw = sec.rows.filter((r) => r.winner === "Deepbox").length;
      const lw = sec.rows.filter((r) => r.winner === "lib").length;
      tableRows += `| ${info.category} | ${dw} | ${lw} | ${info.against} |\n`;
      for (const r of sec.rows) {
        if (r.winner === "Deepbox" && r.speedup > 1.5) {
          const prev = bestPerCat.get(info.category);
          if (!prev || r.speedup > prev.speed) {
            bestPerCat.set(info.category, {
              op: r.operation,
              size: r.size,
              speed: r.speedup,
              cat: info.category,
            });
          }
        }
      }
    }
    const topWins = [...bestPerCat.values()].sort((a, b) => b.speed - a.speed).slice(0, 8);
    let highlights = "";
    for (const w of topWins) {
      highlights += `- **${w.op}** (${w.size}) \u2014 ${w.speed.toFixed(1)}x faster *(${w.cat})*\n`;
    }

    const activeCount = sections.filter((s) => s.rows.length > 0).length;
    let perfSection = `## Performance\n\n`;
    perfSection += `Deepbox is pure TypeScript \u2014 no native addons, no WebAssembly, no C bindings. Every operation runs on V8\u2019s JIT compiler with \`TypedArray\` backing. Despite competing against Python libraries that use hand-tuned C and Fortran backends (BLAS, LAPACK, ATen), Deepbox delivers competitive or superior performance in several areas.\n\n`;
    perfSection += `**${total} head-to-head benchmarks** across ${activeCount} categories, tested on the same machine with identical data sizes and median-based winner selection. Deepbox-only local cases are tracked separately and excluded from the win totals.\n\n`;
    perfSection += `Two win rates are reported. **Overall:** ${totalDs}/${total} (${pct(totalDs)}). **Realized-work** (excludes ${floorExcluded} sub-microsecond lazy-view/spec-build cases that only rewrite shape/stride metadata): ${realizedDs}/${realizedTotal} (${realizedPct(realizedDs)}). The realized-work rate is the fair measure of throughput on operations that actually move data; see [\`benchmarks/RESULTS.md\`](benchmarks/RESULTS.md) for the per-case breakdown.\n\n`;
    perfSection += `| Category | Deepbox Wins | Python Package Wins | Competing Against |\n`;
    perfSection += `| --- | ---: | ---: | --- |\n`;
    perfSection += tableRows;
    perfSection += `| **Total** | **${totalDs}** | **${totalLib}** | |\n\n`;
    perfSection += `### Where Deepbox shines\n\n`;
    perfSection += highlights;
    perfSection += `\n### Context\n\n`;
    perfSection += `Python\u2019s numerical libraries delegate heavy lifting to compiled C/Fortran code (OpenBLAS, MKL, LAPACK). Deepbox implements everything in TypeScript, relying on V8\u2019s TurboFan JIT and \`Float64Array\` for performance. The gap is largest for BLAS-bound operations (matmul, decompositions) and smallest for memory-layout operations (transpose, reshape, indexing) where Deepbox\u2019s lazy-view architecture has an advantage.\n\n`;
    perfSection += `> Run \`npm run bench:all\` to reproduce. Full results in [\`benchmarks/RESULTS.md\`](benchmarks/RESULTS.md).\n`;

    const before = readme.slice(0, perfStart);
    const after = readme.slice(nextH2 + 1);
    readme = `${before}${perfSection}\n${after}`;
    writeFileSync(readmePath, readme);
    console.log(`  \u2713 Updated \u2192 README.md (Performance section)\n`);
  } else {
    console.log(`  \u26a0 Could not locate ## Performance section in README.md\n`);
  }
} else {
  console.log(`  \u26a0 README.md not found\n`);
}
