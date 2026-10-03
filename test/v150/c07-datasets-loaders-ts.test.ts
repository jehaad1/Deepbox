/**
 * v1.5.0 regression tests for src/datasets: loaders, remote CSV, samplers,
 * streaming pipelines, text fetchers, dataset transforms and shared utils.
 * Reference values come from numpy 2.4 / scipy 1.17 / scikit-learn 1.8 / torch 2.12.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import {
  DataValidationError,
  DeepboxError,
  DTypeError,
  InvalidParameterError,
  ShapeError,
} from "../../src/core";
import {
  asyncIterableDataset,
  defaultCollate,
  fetch20Newsgroups,
  fetchCSVDataset,
  fetchIMDB,
  filterDataset,
  iterableDataset,
  loadBreastCancer,
  loadDiabetes,
  loadDigits,
  loadIris,
  loadLinnerud,
  loadWine,
  mapDataset,
  parseCSV,
  randomSplit,
  SequentialSampler,
  Subset,
  SubsetRandomSampler,
  WeightedRandomSampler,
} from "../../src/datasets";
import { shuffleInPlace } from "../../src/datasets/utils";
import { tensor, transpose } from "../../src/ndarray";

afterEach(() => {
  vi.unstubAllGlobals();
});

function sum(values: readonly number[]): number {
  let s = 0;
  for (const v of values) s += v;
  return s;
}

// ─── loaders ─────────────────────────────────────────────────────────────────

describe("loaders: dtype option and real Linnerud", () => {
  it("float64 option returns the exact reference values (sklearn float64)", () => {
    const iris = loadIris({ dtype: "float64" });
    expect(iris.data.dtype).toBe("float64");
    expect(iris.target.dtype).toBe("int32");
    // sklearn load_iris().data.sum() == 2078.7, data[149] == [5.9, 3.0, 5.1, 1.8]
    expect(sum((iris.data.toArray() as number[][]).flat())).toBeCloseTo(2078.7, 10);
    expect((iris.data.toArray() as number[][])[149]).toEqual([5.9, 3, 5.1, 1.8]);

    const diabetes = loadDiabetes({ dtype: "float64" });
    expect(diabetes.target.dtype).toBe("float64");
    expect(sum(diabetes.target.toArray() as number[])).toBe(67243);
    expect((diabetes.data.toArray() as number[][])[0]?.[0]).toBe(0.038075906433423026);
    expect((diabetes.data.toArray() as number[][])[441]?.[9]).toBe(0.0030644094143684884);
  });

  it("keeps the default dtype when no option is given", () => {
    expect(loadIris().data.dtype).toBe("float32");
    expect(loadWine().data.dtype).toBe("float32");
    expect(loadBreastCancer().data.dtype).toBe("float32");
    expect(loadDigits({ dtype: "float64" }).images?.dtype).toBe("float64");
  });

  it("rejects an unsupported dtype", () => {
    // @ts-expect-error invalid dtype on purpose
    expect(() => loadIris({ dtype: "int32" })).toThrow(InvalidParameterError);
  });

  it("loadLinnerud returns the real sklearn data, not synthetic values", () => {
    const l = loadLinnerud({ dtype: "float64" });
    expect(l.data.shape).toEqual([20, 3]);
    expect(l.target.shape).toEqual([20, 3]);
    const data = l.data.toArray() as number[][];
    const target = l.target.toArray() as number[][];
    // sklearn: data.sum(0) == [189, 2911, 1406], target.sum(0) == [3572, 708, 1122]
    for (let j = 0; j < 3; j++) {
      expect(sum(data.map((r) => r[j] as number))).toBe([189, 2911, 1406][j]);
      expect(sum(target.map((r) => r[j] as number))).toBe([3572, 708, 1122][j]);
    }
    expect(target[7]).toEqual([167, 34, 60]);
    expect(l.featureNames).toEqual(["Chins", "Situps", "Jumps"]);
    expect(l.targetNames).toEqual(["Weight", "Waist", "Pulse"]);
    expect(l.description).not.toMatch(/synthetic/i);
  });

  it("repeated loads are independent copies", () => {
    const a = loadLinnerud();
    const b = loadLinnerud();
    (a.data.data as Float32Array)[0] = 999;
    expect(b.data.at(0, 0)).toBe(5);
  });
});

// ─── remote CSV ──────────────────────────────────────────────────────────────

describe("parseCSV: strict validation", () => {
  it("rejects empty cells instead of silently reading them as 0", () => {
    expect(() => parseCSV("a,b,y\n1,,3\n4,5,6\n")).toThrow(DataValidationError);
    expect(() => parseCSV("a,b,y\n1,,3\n4,5,6\n")).toThrow(/Missing value at line 2, column 2/);
  });

  it("rejects rows whose cell count differs from the header", () => {
    expect(() => parseCSV("a,b,y\n1,2,3\n4,5\n")).toThrow(/Line 3 has 2 cells; expected 3/);
    expect(() => parseCSV("a,b,y\n1,2,3,4\n")).toThrow(/Line 2 has 4 cells; expected 3/);
    expect(() => parseCSV("1,2,3\n4,5\n", { header: false })).toThrow(/Line 2 has 2 cells/);
  });

  it("reports original line numbers even with blank lines and CRLF", () => {
    expect(() => parseCSV("a,y\r\n\r\n1,2\r\n3,x\r\n")).toThrow(/line 4, column 2/);
  });

  it("only accepts decimal numbers", () => {
    expect(() => parseCSV("a,y\n0x10,1\n")).toThrow(/Non-numeric/);
    expect(() => parseCSV("a,y\nInfinity,1\n")).toThrow(/Non-numeric/);
    expect(() => parseCSV("a,y\nNaN,1\n")).toThrow(/Non-numeric/);
    expect(() => parseCSV("a,y\n1e999,1\n")).toThrow(/finite number range/);
    const ds = parseCSV("a,y\n+1.5,-2\n.5,1e-3\n5.,2E2\n");
    expect(ds.data.toArray()).toEqual([[1.5], [0.5], [5]]);
    expect((ds.target.toArray() as number[])[2]).toBe(200);
  });

  it("supports quoted fields with embedded line breaks in the header", () => {
    const ds = parseCSV('"a\nb",y\n1,2\n3,4\n');
    expect(ds.featureNames).toEqual(["a\nb"]);
    expect(ds.data.shape).toEqual([2, 1]);
    expect(() => parseCSV('a,"y\n1,2\n')).toThrow(/Unterminated quoted field starting at line 1/);
  });

  it("handles lone carriage-return line endings and a BOM", () => {
    const ds = parseCSV("﻿a,y\r1,2\r3,4\r");
    expect(ds.featureNames).toEqual(["a"]);
    expect(ds.data.shape).toEqual([2, 1]);
  });

  it("names empty header cells feature_<i>", () => {
    const ds = parseCSV(",b,y\n1,2,3\n");
    expect(ds.featureNames).toEqual(["feature_0", "b"]);
  });

  it("validates options", () => {
    expect(() => parseCSV("a,y\n1,2\n", { targetColumn: 0.5 })).toThrow(InvalidParameterError);
    expect(() => parseCSV("a,y\n1,2\n", { targetColumn: -1 })).toThrow(/targetColumn/);
    expect(() => parseCSV("a,y\n1,2\n", { separator: "||" })).toThrow(InvalidParameterError);
    expect(() => parseCSV("a,y\n1,2\n", { separator: "" })).toThrow(InvalidParameterError);
    expect(() => parseCSV("a,y\n1,2\n", { separator: '"' })).toThrow(InvalidParameterError);
    // @ts-expect-error invalid type on purpose
    expect(() => parseCSV(123)).toThrow(InvalidParameterError);
  });

  it("reports a header-only CSV as having no data rows", () => {
    expect(() => parseCSV("a,b\n")).toThrow(/no data rows/);
  });

  it("supports tab separators and a custom target column", () => {
    const ds = parseCSV("x\ty\tz\n1\t2\t3\n4\t5\t6\n", { separator: "\t", targetColumn: 1 });
    expect(ds.featureNames).toEqual(["x", "z"]);
    expect(ds.targetName).toBe("y");
    expect(ds.target.toArray()).toEqual([2, 5]);
    expect(ds.data.toArray()).toEqual([
      [1, 3],
      [4, 6],
    ]);
  });
});

describe("fetchCSVDataset: option validation and signals", () => {
  const okResponse = (body: string) => new Response(body, { status: 200 });

  it("rejects NaN, negative or non-numeric timeouts before fetching", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    for (const timeout of [Number.NaN, -1, Number.POSITIVE_INFINITY]) {
      await expect(fetchCSVDataset({ url: "http://x/data.csv", timeout })).rejects.toThrow(
        InvalidParameterError
      );
    }
    await expect(fetchCSVDataset({ url: "http://x/data.csv", separator: "ab" })).rejects.toThrow(
      InvalidParameterError
    );
    await expect(fetchCSVDataset({ url: "http://x/data.csv", targetColumn: 1.5 })).rejects.toThrow(
      InvalidParameterError
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("applies the timeout even when an abortSignal is supplied", async () => {
    vi.stubGlobal("fetch", (_url: string, init: { signal?: AbortSignal }) => {
      return new Promise((_resolve, reject) => {
        init.signal?.addEventListener("abort", () => reject(init.signal?.reason));
      });
    });
    const controller = new AbortController();
    await expect(
      fetchCSVDataset({ url: "http://x/slow.csv", timeout: 30, abortSignal: controller.signal })
    ).rejects.toThrow(/Timed out/);
  });

  it("aborts when the supplied signal fires before the timeout", async () => {
    vi.stubGlobal("fetch", (_url: string, init: { signal?: AbortSignal }) => {
      return new Promise((_resolve, reject) => {
        init.signal?.addEventListener("abort", () => reject(init.signal?.reason));
      });
    });
    const controller = new AbortController();
    const pending = fetchCSVDataset({
      url: "http://x/slow.csv",
      timeout: 5000,
      abortSignal: controller.signal,
    });
    controller.abort();
    await expect(pending).rejects.toThrow(/aborted/);
  });

  it("does not abort immediately for a timeout above the 32-bit timer limit", async () => {
    vi.stubGlobal("fetch", (_url: string, init: { signal?: AbortSignal }) => {
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => resolve(okResponse("a,y\n1,2\n")), 30);
        init.signal?.addEventListener("abort", () => {
          clearTimeout(timer);
          reject(init.signal?.reason);
        });
      });
    });
    const ds = await fetchCSVDataset({ url: "http://x/d.csv", timeout: 5e9 });
    expect(ds.target.toArray()).toEqual([2]);
  });

  it("surfaces CSV validation errors as DataValidationError", async () => {
    vi.stubGlobal("fetch", async () => okResponse("a,y\n1,\n"));
    await expect(fetchCSVDataset({ url: "http://x/d.csv" })).rejects.toThrow(DataValidationError);
  });

  it("parses a normal response", async () => {
    vi.stubGlobal("fetch", async () => okResponse("a,b,y\n1,2,3\n4,5,6\n"));
    const ds = await fetchCSVDataset({ url: "http://x/d.csv", timeout: 0 });
    expect(ds.data.shape).toEqual([2, 2]);
    expect(ds.target.toArray()).toEqual([3, 6]);
  });
});

// ─── samplers ────────────────────────────────────────────────────────────────

describe("samplers", () => {
  it("WeightedRandomSampler without replacement cannot exceed the positive weights (numpy parity)", () => {
    // numpy: choice(4, size=3, replace=False, p=[.5,.5,0,0]) -> ValueError
    expect(
      () => new WeightedRandomSampler([0.5, 0.5, 0, 0], { numSamples: 3, replacement: false })
    ).toThrow(/positive weights/);
    const ok = new WeightedRandomSampler([0.5, 0.5, 0, 0], { numSamples: 2, replacement: false });
    expect([...ok].sort()).toEqual([0, 1]);
  });

  it("never draws zero-weight indices", () => {
    const withRepl = new WeightedRandomSampler([0, 3, 0, 1, 0], { numSamples: 2000, seed: 5 });
    const counts = [0, 0, 0, 0, 0];
    for (const i of withRepl) counts[i] = (counts[i] as number) + 1;
    expect(counts[0]).toBe(0);
    expect(counts[2]).toBe(0);
    expect(counts[4]).toBe(0);
    // Expected frequencies 0.75 / 0.25.
    expect(counts[1] as number).toBeGreaterThan(1400);
    expect(counts[1] as number).toBeLessThan(1600);

    const without = new WeightedRandomSampler([0, 3, 0, 1, 0], {
      numSamples: 2,
      replacement: false,
      seed: 1,
    });
    expect([...without].sort()).toEqual([1, 3]);
  });

  it("without replacement returns every positive index exactly once, tiny weights included", () => {
    const w = [1e-12, 1, 1e12, 1e-300, 5];
    for (let seed = 0; seed < 50; seed++) {
      const s = new WeightedRandomSampler(w, { numSamples: 5, replacement: false, seed });
      expect([...s].sort()).toEqual([0, 1, 2, 3, 4]);
    }
  });

  it("without replacement matches the successive-draw distribution", () => {
    // P(first draw = i) = w_i / sum(w).
    const first = [0, 0, 0];
    for (let seed = 0; seed < 3000; seed++) {
      const s = new WeightedRandomSampler([1, 2, 3], { numSamples: 1, replacement: false, seed });
      const [idx] = [...s];
      first[idx as number] = (first[idx as number] as number) + 1;
    }
    expect(first[0] as number).toBeGreaterThan(3000 / 6 - 120);
    expect(first[0] as number).toBeLessThan(3000 / 6 + 120);
    expect(first[2] as number).toBeGreaterThan(3000 / 2 - 150);
    expect(first[2] as number).toBeLessThan(3000 / 2 + 150);
  });

  it("rejects weights whose sum overflows", () => {
    expect(() => new WeightedRandomSampler([1e308, 1e308])).toThrow(InvalidParameterError);
  });

  it("validates numSamples, replacement and seed", () => {
    expect(() => new WeightedRandomSampler([1, 1], { numSamples: 0 })).toThrow(
      InvalidParameterError
    );
    expect(() => new WeightedRandomSampler([1, 1], { numSamples: 1.5 })).toThrow(
      InvalidParameterError
    );
    // @ts-expect-error invalid type on purpose
    expect(() => new WeightedRandomSampler([1, 1], { replacement: "yes" })).toThrow(
      InvalidParameterError
    );
    expect(() => new WeightedRandomSampler([1, 1], { seed: 1.5 })).toThrow(InvalidParameterError);
    expect(() => new SubsetRandomSampler([1, 2], { seed: Number.NaN })).toThrow(
      InvalidParameterError
    );
  });

  it("copies the input arrays so later mutation does not change the sampler", () => {
    const weights = [1, 1, 1];
    const w = new WeightedRandomSampler(weights, { numSamples: 3, seed: 1 });
    const before = [...w];
    weights[0] = -5;
    weights.length = 0;
    expect([...w]).toEqual(before);

    const indices = [3, 4, 5];
    const s = new SubsetRandomSampler(indices, { seed: 2 });
    indices.push(99);
    indices[0] = -1;
    expect(s.length).toBe(3);
    expect([...s].sort()).toEqual([3, 4, 5]);
  });

  it("accepts typed arrays", () => {
    const s = new SubsetRandomSampler(new Int32Array([4, 5, 6]), { seed: 1 });
    expect([...s].sort()).toEqual([4, 5, 6]);
    const w = new WeightedRandomSampler(new Float64Array([0, 1]), { numSamples: 4 });
    expect([...w]).toEqual([1, 1, 1, 1]);
  });

  it("SequentialSampler rejects unsafe lengths", () => {
    expect(() => new SequentialSampler(2 ** 60)).toThrow(InvalidParameterError);
    expect(() => new SequentialSampler(1.5)).toThrow(InvalidParameterError);
  });

  it("seeded samplers repeat; unseeded ones are permutations", () => {
    const s = new SubsetRandomSampler([0, 1, 2, 3, 4, 5, 6, 7], { seed: 11 });
    expect([...s]).toEqual([...s]);
    const u = new SubsetRandomSampler([0, 1, 2, 3, 4, 5, 6, 7]);
    expect([...u].sort()).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });
});

// ─── utils ───────────────────────────────────────────────────────────────────

describe("shuffleInPlace", () => {
  it("shuffles arrays that contain undefined without throwing", () => {
    const arr: (number | undefined)[] = [undefined, 1, undefined, 2, 3];
    let seed = 7;
    const rng = () => {
      seed = (seed * 16807) % 2147483647;
      return seed / 2147483647;
    };
    shuffleInPlace(arr, rng);
    expect(arr.filter((v) => v === undefined)).toHaveLength(2);
    expect(arr.filter((v) => v !== undefined).sort()).toEqual([1, 2, 3]);
  });
});

// ─── streaming ───────────────────────────────────────────────────────────────

describe("streaming", () => {
  it("prefetch does not leak unhandled rejections when a later read fails", async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown) => unhandled.push(reason);
    process.on("unhandledRejection", onUnhandled);
    try {
      const ds = asyncIterableDataset<number>(async function* () {
        yield 0;
        throw new Error("boom");
      }).prefetch(3);
      const seen: number[] = [];
      let caught: unknown;
      try {
        for await (const v of ds) {
          seen.push(v);
          // Hold the consumer while the failing read settles.
          await new Promise((r) => setTimeout(r, 40));
        }
      } catch (err) {
        caught = err;
      }
      expect(seen).toEqual([0]);
      expect((caught as Error).message).toBe("boom");
      await new Promise((r) => setTimeout(r, 20));
      expect(unhandled).toEqual([]);
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });

  it("defaultCollate rejects mixed labeled/unlabeled samples", () => {
    expect(() =>
      defaultCollate([
        [[1, 2], 0],
        [3, 4],
      ] as never)
    ).toThrow(InvalidParameterError);
    expect(() =>
      defaultCollate([
        [3, 4],
        [[1, 2], 0],
      ] as never)
    ).toThrow(InvalidParameterError);
  });

  it("defaultCollate rejects ragged features and labels", () => {
    expect(() =>
      defaultCollate([
        [[1, 2], 0],
        [[3], 1],
      ])
    ).toThrow(/has 1 features; expected 2/);
    expect(() => defaultCollate([[1, 2], [3]])).toThrow(/has 1 features; expected 2/);
    expect(() =>
      defaultCollate([
        [[1], [1, 2]],
        [[2], [3]],
      ])
    ).toThrow(InvalidParameterError);
    expect(() =>
      defaultCollate([
        [[1], 0],
        [[2], [3]],
      ] as never)
    ).toThrow(InvalidParameterError);
  });

  it("defaultCollate keeps integer labels exact outside the int32 range", () => {
    const [, y] = defaultCollate([
      [[1], 3_000_000_000],
      [[2], 1],
    ]);
    expect(y?.dtype).toBe("float64");
    expect(y?.toArray()).toEqual([3_000_000_000, 1]);
    const [, small] = defaultCollate([
      [[1], 2],
      [[2], -1],
    ]);
    expect(small?.dtype).toBe("int32");
    const [, frac] = defaultCollate([
      [[1], 2.5],
      [[2], 1],
    ]);
    expect(frac?.dtype).toBe("float32");
  });

  it("defaultCollate accepts typed-array features but not DataView", () => {
    const [x] = defaultCollate([new Float32Array([1, 2]), new Float32Array([3, 4])] as never);
    expect(x.shape).toEqual([2, 2]);
    const dv = new DataView(new ArrayBuffer(8));
    expect(() => defaultCollate([dv as never])).toThrow();
  });

  it("validates seed, collate, map and dropLast arguments", () => {
    const ds = iterableDataset(function* () {
      yield [1, 2];
    });
    expect(() => ds.shuffle(4, 1.5)).toThrow(InvalidParameterError);
    expect(() => ds.shuffle(2 ** 60)).toThrow(InvalidParameterError);
    // @ts-expect-error invalid type on purpose
    expect(() => ds.map(5)).toThrow(InvalidParameterError);
    // @ts-expect-error invalid type on purpose
    expect(() => ds.batch(2, 5)).toThrow(InvalidParameterError);
    // @ts-expect-error invalid type on purpose
    expect(() => ds.batch(2, undefined, { dropLast: "yes" })).toThrow(InvalidParameterError);
    expect(() => ds.prefetch(0)).toThrow(InvalidParameterError);
  });

  it("seeded shuffle repeats the same order on every pass", () => {
    const ds = iterableDataset(function* () {
      for (let i = 0; i < 30; i++) yield i;
    }).shuffle(8, 3);
    expect([...ds]).toEqual([...ds]);
    expect([...ds].sort((a, b) => a - b)).toEqual(Array.from({ length: 30 }, (_, i) => i));
  });
});

// ─── transforms ──────────────────────────────────────────────────────────────

describe("transforms", () => {
  it("filterDataset with no match keeps the feature axes and dtypes", () => {
    const iris = loadIris();
    const empty = filterDataset(iris, () => false);
    expect(empty.data.shape).toEqual([0, 4]);
    expect(empty.target.shape).toEqual([0]);
    expect(empty.data.dtype).toBe(iris.data.dtype);
    expect(empty.target.dtype).toBe("int32");
  });

  it("Subset copies indices and metadata arrays", () => {
    const iris = loadIris();
    const indices = [0, 1];
    const sub = new Subset(iris, indices);
    indices.push(2);
    indices[0] = 100;
    expect(sub.indices).toEqual([0, 1]);
    iris.featureNames.push("extra");
    expect(sub.featureNames).toHaveLength(4);
    expect(sub.targetNames).toEqual(["setosa", "versicolor", "virginica"]);
  });

  it("Subset and filterDataset keep images", () => {
    const digits = loadDigits();
    const sub = new Subset(digits, [5, 0]);
    expect(sub.images?.shape).toEqual([2, 8, 8]);
    expect(sub.images?.at(0, 0, 3)).toBe(digits.images?.at(5, 0, 3));
    const f = filterDataset(digits, (_row, t) => t === 3);
    expect(f.images?.shape[0]).toBe(f.data.shape[0]);
  });

  it("Subset rejects datasets whose data and target disagree on sample count", () => {
    const bad = { ...loadIris(), target: tensor([0, 1, 2], { dtype: "int32" }) };
    expect(() => new Subset(bad, [0])).toThrow(ShapeError);
    expect(() => new Subset(loadIris(), 3 as never)).toThrow(InvalidParameterError);
  });

  it("randomSplit supports fractional lengths like torch random_split", () => {
    const iris = loadIris();
    // torch: random_split(range(150), [0.8, 0.2]) -> [120, 30]
    expect(randomSplit(iris, [0.8, 0.2], 1).map((s) => s.data.shape[0])).toEqual([120, 30]);
    const sub = (n: number) => ({
      ...iris,
      data: tensor(Array.from({ length: n }, (_, i) => [i])),
      target: tensor(
        Array.from({ length: n }, (_, i) => i),
        { dtype: "int32" as const }
      ),
    });
    // torch: n=10 [0.55,0.45] -> [6,4]; [0.33,0.33,0.34] -> [4,3,3]; n=7 [0.5,0.5] -> [4,3]
    expect(randomSplit(sub(10), [0.55, 0.45], 1).map((s) => s.indices.length)).toEqual([6, 4]);
    expect(randomSplit(sub(10), [0.33, 0.33, 0.34], 1).map((s) => s.indices.length)).toEqual([
      4, 3, 3,
    ]);
    expect(randomSplit(sub(7), [0.5, 0.5], 1).map((s) => s.indices.length)).toEqual([4, 3]);
    expect(randomSplit(sub(10), [0.1, 0.1, 0.8], 1).map((s) => s.indices.length)).toEqual([
      1, 1, 8,
    ]);
    // Fractions are a partition of all indices.
    const all = randomSplit(sub(10), [0.55, 0.45], 9).flatMap((s) => [...s.indices]);
    expect(all.sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it("randomSplit treats lengths that sum to 1 as fractions (torch parity)", () => {
    const iris = loadIris();
    // torch: random_split(range(150), [1.0]) -> [150]
    expect(randomSplit(iris, [1], 1).map((s) => s.data.shape[0])).toEqual([150]);
    expect(randomSplit(iris, [0, 1], 1).map((s) => s.data.shape[0])).toEqual([0, 150]);
  });

  it("randomSplit rejects fractions that do not sum to 1 and bad seeds", () => {
    const iris = loadIris();
    expect(() => randomSplit(iris, [0.5, 0.4], 1)).toThrow(InvalidParameterError);
    expect(() => randomSplit(iris, [0.5, 0.5, 0.5], 1)).toThrow(InvalidParameterError);
    expect(() => randomSplit(iris, [1.5, -0.5], 1)).toThrow(InvalidParameterError);
    expect(() => randomSplit(iris, [150], 1.5)).toThrow(InvalidParameterError);
    expect(() => randomSplit(iris, [100, 30])).toThrow(/Sum of lengths/);
  });

  it("randomSplit allows empty subsets", () => {
    const [a, b] = randomSplit(loadIris(), [150, 0], 3);
    expect(a?.data.shape).toEqual([150, 4]);
    expect(b?.data.shape).toEqual([0, 4]);
  });

  it("mapDataset keeps float64 data and does not truncate fractional targets", () => {
    const iris = loadIris({ dtype: "float64" });
    const mapped = mapDataset(iris, (data, target) => ({
      data: data.map((v) => v / 3),
      target: target + 0.5,
    }));
    expect(mapped.data.dtype).toBe("float64");
    expect(mapped.data.at(0, 0)).toBe(5.1 / 3);
    // An int32 target mapped to fractional values must not be silently truncated.
    expect(mapped.target.dtype).toBe("float32");
    expect(mapped.target.at(0)).toBe(0.5);
    expect(mapped.target.at(149)).toBe(2.5);
    // Integer-valued results keep the original dtype.
    const kept = mapDataset(iris, (data, target) => ({ data, target: target + 1 }));
    expect(kept.target.dtype).toBe("int32");
    expect(kept.target.at(149)).toBe(3);
  });

  it("mapDataset can change the number of features and rejects ragged output", () => {
    const iris = loadIris();
    const wider = mapDataset(iris, (data, target) => ({ data: [...data, data[0] ?? 0], target }));
    expect(wider.data.shape).toEqual([150, 5]);
    expect(() =>
      mapDataset(iris, (data, target) => ({
        data: target === 0 ? data : data.slice(1),
        target,
      }))
    ).toThrow(ShapeError);
    // @ts-expect-error invalid return on purpose
    expect(() => mapDataset(iris, () => ({ data: 5, target: 1 }))).toThrow(InvalidParameterError);
    // @ts-expect-error invalid return on purpose
    expect(() => mapDataset(iris, () => null)).toThrow(InvalidParameterError);
  });

  it("mapDataset and filterDataset pass flattened samples for N-d data and reject multi-output targets", () => {
    const digits = loadDigits();
    const imageDs = { ...digits, data: digits.images as NonNullable<typeof digits.images> };
    let width = -1;
    const mapped = mapDataset(imageDs, (data, target) => {
      width = data.length;
      return { data, target };
    });
    expect(width).toBe(64);
    expect(mapped.data.shape).toEqual([1797, 64]);

    expect(() => mapDataset(loadLinnerud(), (d, t) => ({ data: d, target: t }))).toThrow(
      ShapeError
    );
    expect(() => filterDataset(loadLinnerud(), () => true)).toThrow(ShapeError);
  });

  it("mapDataset and filterDataset work on non-contiguous (transposed/sliced) data", () => {
    const base = tensor([
      [1, 10],
      [2, 20],
      [3, 30],
    ]);
    const view = base.slice({}, { start: 1, end: 2 }); // second column only, shape [3, 1]
    const ds = {
      data: view,
      target: tensor([0, 1, 0], { dtype: "int32" as const }),
      featureNames: ["x"],
      description: "view",
    };
    const seen: number[][] = [];
    const filtered = filterDataset(ds, (row) => {
      seen.push(row);
      return row[0] === 20 || row[0] === 30;
    });
    expect(seen).toEqual([[10], [20], [30]]);
    expect(filtered.data.toArray()).toEqual([[20], [30]]);
    const mapped = mapDataset(ds, (row, target) => ({ data: [(row[0] as number) + 1], target }));
    expect(mapped.data.toArray()).toEqual([[11], [21], [31]]);
  });

  it("reads transposed views in logical order", () => {
    const t = transpose(
      tensor([
        [1, 2, 3],
        [10, 20, 30],
      ])
    ); // logical rows: [1, 10], [2, 20], [3, 30]
    const ds = {
      data: t,
      target: tensor([0, 1, 2], { dtype: "int32" as const }),
      featureNames: ["a", "b"],
      description: "t",
    };
    const mapped = mapDataset(ds, (row, target) => ({ data: row, target }));
    expect(mapped.data.toArray()).toEqual([
      [1, 10],
      [2, 20],
      [3, 30],
    ]);
    expect(filterDataset(ds, (row) => row[0] !== 2).data.toArray()).toEqual([
      [1, 10],
      [3, 30],
    ]);
  });

  it("mapDataset on an empty dataset keeps the feature width", () => {
    const empty = filterDataset(loadIris(), () => false);
    const mapped = mapDataset(empty, (data, target) => ({ data, target }));
    expect(mapped.data.shape).toEqual([0, 4]);
  });

  it("rejects string tensors with a typed error", () => {
    const ds = {
      data: tensor([["a"], ["b"]] as never),
      target: tensor([0, 1], { dtype: "int32" as const }),
      featureNames: ["x"],
      description: "s",
    };
    expect(() => mapDataset(ds, (d, t) => ({ data: d, target: t }))).toThrow(DTypeError);
  });
});

// ─── text ────────────────────────────────────────────────────────────────────

describe("text fetchers", () => {
  // The default location is now the official archive; the JSON layout checks use a mirror URL.
  const MIRROR = "https://host/";
  const jsonResponse = (body: unknown) => new Response(JSON.stringify(body), { status: 200 });

  it("returns an int32 target and appends the file name to a base URL without trailing slash", async () => {
    const calls: string[] = [];
    vi.stubGlobal("fetch", async (url: string) => {
      calls.push(url);
      return jsonResponse({
        data: ["a", "b", "c"],
        target: [0, 1, 1],
        target_names: ["x", "y"],
      });
    });
    const news = await fetch20Newsgroups({ baseUrl: "https://host/data", subset: "train" });
    expect(calls[0]).toBe("https://host/data/20newsgroups_train.json");
    expect(news.target.dtype).toBe("int32");
    expect(news.nClasses).toBe(2);
    expect(news.isSynthetic).toBe(false);

    const imdb = await fetchIMDB({ baseUrl: "https://host/data/", maxSamples: 2 });
    expect(calls[1]).toBe("https://host/data/imdb_all.json");
    expect(imdb.texts).toEqual(["a", "b"]);
    expect(imdb.target.dtype).toBe("int32");
    expect(imdb.target.toArray()).toEqual([0, 1]);
  });

  it("accepts fractional and very large timeouts", async () => {
    vi.stubGlobal("fetch", async () =>
      jsonResponse({ data: ["a", "b"], target: [0, 1], target_names: ["x", "y"] })
    );
    const a = await fetchIMDB({ baseUrl: "https://host/", timeout: 1500.5 });
    expect(a.texts).toHaveLength(2);
    const b = await fetchIMDB({ baseUrl: "https://host/", timeout: 1e12 });
    expect(b.texts).toHaveLength(2);
  });

  it("rejects maxSamples: 0 and invalid subsets instead of ignoring them", async () => {
    await expect(fetchIMDB({ maxSamples: 0 })).rejects.toThrow(InvalidParameterError);
    await expect(fetch20Newsgroups({ maxSamples: -3 })).rejects.toThrow(InvalidParameterError);
    // @ts-expect-error invalid subset on purpose
    await expect(fetchIMDB({ subset: "val" })).rejects.toThrow(InvalidParameterError);
    await expect(fetchIMDB({ baseUrl: "" })).rejects.toThrow(InvalidParameterError);
    await expect(fetchIMDB({ timeout: -1 })).rejects.toThrow(InvalidParameterError);
  });

  it("includes the failure reason in the error message", async () => {
    vi.stubGlobal(
      "fetch",
      async () => new Response("nope", { status: 404, statusText: "Not Found" })
    );
    await expect(fetchIMDB({ baseUrl: "https://host/" })).rejects.toThrow(
      /Failed to fetch IMDB dataset from https:\/\/host\/imdb_all\.json: HTTP 404 Not Found/
    );
    vi.stubGlobal("fetch", async () => {
      throw new TypeError("network down");
    });
    await expect(fetch20Newsgroups()).rejects.toThrow(/network down/);
  });

  it("rejects malformed JSON layouts rather than returning mismatched data", async () => {
    vi.stubGlobal("fetch", async () =>
      jsonResponse({ data: ["a", "b"], target: [0], target_names: ["x"] })
    );
    await expect(fetch20Newsgroups({ baseUrl: MIRROR })).rejects.toThrow(
      /"data" has 2 entries but "target" has 1/
    );

    vi.stubGlobal("fetch", async () =>
      jsonResponse({ data: ["a"], target: [5], target_names: ["x", "y"] })
    );
    await expect(fetch20Newsgroups({ baseUrl: MIRROR })).rejects.toThrow(
      /must be an integer in \[0, 1\]/
    );

    vi.stubGlobal("fetch", async () => jsonResponse({ data: ["a"], target: [2] }));
    await expect(fetchIMDB({ baseUrl: MIRROR })).rejects.toThrow(/\[0, 1\]/);

    vi.stubGlobal("fetch", async () =>
      jsonResponse({ data: [1], target: [0], target_names: ["x"] })
    );
    await expect(fetch20Newsgroups({ baseUrl: MIRROR })).rejects.toThrow(/is not a string/);

    vi.stubGlobal("fetch", async () => jsonResponse({ data: ["a"], target: [0] }));
    await expect(fetch20Newsgroups({ baseUrl: MIRROR })).rejects.toThrow(/target_names/);
  });

  it("falls back to synthetic data with int32 labels when allowed", async () => {
    vi.stubGlobal("fetch", async () => new Response("", { status: 500, statusText: "Err" }));
    const imdb = await fetchIMDB({ maxSamples: 5, allowSyntheticFallback: true });
    expect(imdb.isSynthetic).toBe(true);
    expect(imdb.target.dtype).toBe("int32");
    expect(imdb.texts).toHaveLength(5);
    const news = await fetch20Newsgroups({ maxSamples: 40, allowSyntheticFallback: true });
    expect(news.target.dtype).toBe("int32");
    expect(news.texts).toHaveLength(40);
    expect(news.classNames).toHaveLength(20);
    // Returned class names are a fresh array each call.
    news.classNames.push("x");
    const again = await fetch20Newsgroups({ maxSamples: 40, allowSyntheticFallback: true });
    expect(again.classNames).toHaveLength(20);
  });

  it("errors are DeepboxError instances", async () => {
    vi.stubGlobal("fetch", async () => new Response("", { status: 404, statusText: "NF" }));
    await expect(fetchIMDB()).rejects.toThrow(DeepboxError);
  });
});
