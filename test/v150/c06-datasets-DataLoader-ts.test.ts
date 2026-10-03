/**
 * v1.5.0 regression tests for src/datasets: DataLoader, synthetic generators,
 * image/Kaggle fetchers and the bundled reference data tables.
 * Reference values come from numpy 2.4 / scikit-learn 1.8.
 */

import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gzipSync } from "node:zlib";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DeepboxError, IndexError, InvalidParameterError } from "../../src/core";
import {
  DataLoader,
  fetchCIFAR10,
  fetchKaggleDataset,
  fetchKaggleDatasetInfo,
  fetchMNIST,
  loadBreastCancer,
  makeBiclusters,
  makeBlobs,
  makeCheckerboard,
  makeClassification,
  makeFriedman1,
  makeFriedman2,
  makeFriedman3,
  makeGaussianQuantiles,
  makeLowRankMatrix,
  makeMoons,
  makeRegression,
  makeSCurve,
  makeSPDMatrix,
  makeSparseUncorrelated,
  makeSwissRoll,
  readKaggleCredentials,
  SequentialSampler,
  SubsetRandomSampler,
  searchKaggleDatasets,
  WeightedRandomSampler,
} from "../../src/datasets";
import {
  BREAST_CANCER_COLS,
  BREAST_CANCER_DATA,
  BREAST_CANCER_ROWS,
  BREAST_CANCER_TARGET,
} from "../../src/datasets/data/breast-cancer.data";
import {
  DIABETES_COLS,
  DIABETES_DATA,
  DIABETES_ROWS,
  DIABETES_TARGET,
} from "../../src/datasets/data/diabetes.data";
import {
  DIGITS_COLS,
  DIGITS_ROWS,
  decodeDigitsData,
  decodeDigitsTarget,
} from "../../src/datasets/data/digits.data";
import { IRIS_DATA, IRIS_TARGET } from "../../src/datasets/data/iris.data";
import { WINE_DATA, WINE_TARGET } from "../../src/datasets/data/wine.data";
import { eigvalsh, svdvals } from "../../src/linalg";
import { tensor, transpose } from "../../src/ndarray";

const f64 = (t: { data: unknown }): number[] => Array.from(t.data as Float64Array);

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

// ─── Generators ──────────────────────────────────────────────────────────────

describe("makeFriedman2 / makeFriedman3 feature ranges", () => {
  it("draws x1 from [40*pi, 560*pi), not [40*pi, 600*pi)", () => {
    for (const make of [makeFriedman2, makeFriedman3]) {
      const [X] = make({ nSamples: 20000, randomState: 3 });
      const d = f64(X);
      let min = Number.POSITIVE_INFINITY;
      let max = Number.NEGATIVE_INFINITY;
      for (let i = 1; i < d.length; i += 4) {
        min = Math.min(min, d[i] as number);
        max = Math.max(max, d[i] as number);
      }
      expect(min).toBeGreaterThanOrEqual(40 * Math.PI);
      expect(max).toBeLessThan(560 * Math.PI);
      // 20000 uniform draws get within 0.1% of the upper end.
      expect(max).toBeGreaterThan(559 * Math.PI);
    }
  });

  it("matches the reference formula without noise", () => {
    const [X, y] = makeFriedman2({ nSamples: 5, randomState: 1 });
    const x = f64(X);
    const yv = f64(y);
    for (let i = 0; i < 5; i++) {
      const [x0, x1, x2, x3] = x.slice(i * 4, i * 4 + 4) as [number, number, number, number];
      expect(yv[i]).toBeCloseTo(Math.sqrt(x0 ** 2 + (x1 * x2 - 1 / (x1 * x3)) ** 2), 9);
    }
    // numpy: sqrt(50^2 + (100*0.5 - 1/(100*5))^2) = 70.7092639192348
    expect(Math.sqrt(50 ** 2 + (100 * 0.5 - 1 / (100 * 5)) ** 2)).toBeCloseTo(70.7092639192348, 12);
  });
});

describe("noise validation on generators", () => {
  it("rejects negative and non-finite noise instead of ignoring it", () => {
    const makers = [
      (noise: number) => makeFriedman1({ noise }),
      (noise: number) => makeFriedman2({ noise }),
      (noise: number) => makeFriedman3({ noise }),
      (noise: number) => makeSwissRoll({ noise }),
      (noise: number) => makeSCurve({ noise }),
      (noise: number) => makeBiclusters({ shape: [5, 5], noise }),
      (noise: number) => makeCheckerboard({ shape: [6, 6], nClusters: [2, 2], noise }),
    ];
    for (const make of makers) {
      expect(() => make(-1)).toThrow(InvalidParameterError);
      expect(() => make(Number.NaN)).toThrow(InvalidParameterError);
      expect(() => make(Number.POSITIVE_INFINITY)).toThrow(InvalidParameterError);
    }
  });
});

describe("makeSparseUncorrelated", () => {
  it("uses y ~ N(x0 + 2 x1 - 2 x2 - 1.5 x3, 1) like scikit-learn", () => {
    const n = 20000;
    const [X, y] = makeSparseUncorrelated({ nSamples: n, nFeatures: 10, randomState: 5 });
    const x = f64(X);
    const yv = f64(y);
    let sum = 0;
    let sumSq = 0;
    let cov4 = 0;
    for (let i = 0; i < n; i++) {
      const r = x.slice(i * 10, i * 10 + 4) as number[];
      const resid =
        (yv[i] as number) -
        ((r[0] as number) + 2 * (r[1] as number) - 2 * (r[2] as number) - 1.5 * (r[3] as number));
      sum += resid;
      sumSq += resid * resid;
      cov4 += resid * (x[i * 10 + 4] as number);
    }
    expect(Math.abs(sum / n)).toBeLessThan(0.03);
    expect(sumSq / n).toBeGreaterThan(0.95);
    expect(sumSq / n).toBeLessThan(1.05);
    expect(Math.abs(cov4 / n)).toBeLessThan(0.03);
  });

  it("requires at least four features", () => {
    expect(() => makeSparseUncorrelated({ nFeatures: 3 })).toThrow(InvalidParameterError);
    expect(makeSparseUncorrelated({ nSamples: 4, nFeatures: 4 })[0].shape).toEqual([4, 4]);
  });
});

describe("makeLowRankMatrix", () => {
  it("has the documented singular value spectrum", () => {
    // numpy: (1-ts) * exp(-(i/r)^2) + ts * exp(-0.1 i / r), i < min(ns, nf)
    const expected = [1.0, 0.9310277086481878, 0.7883436867307863, 0.6363584296037009];
    const M = makeLowRankMatrix({
      nSamples: 6,
      nFeatures: 4,
      effectiveRank: 3,
      tailStrength: 0.5,
      randomState: 11,
    });
    const s = f64(svdvals(M));
    expect(s).toHaveLength(4);
    for (let i = 0; i < 4; i++) expect(s[i]).toBeCloseTo(expected[i] as number, 10);
  });

  it("supports wide matrices and a custom tail strength", () => {
    const expected = [
      1.0, 0.8219079434287322, 0.5021189353875717, 0.2942264125276627, 0.2184194174360461,
    ];
    const M = makeLowRankMatrix({
      nSamples: 5,
      nFeatures: 9,
      effectiveRank: 2,
      tailStrength: 0.25,
      randomState: 2,
    });
    const s = f64(svdvals(M));
    for (let i = 0; i < 5; i++) expect(s[i]).toBeCloseTo(expected[i] as number, 10);
  });

  it("validates tailStrength and is reproducible", () => {
    expect(() => makeLowRankMatrix({ tailStrength: 1.5 })).toThrow(InvalidParameterError);
    expect(() => makeLowRankMatrix({ tailStrength: -0.1 })).toThrow(InvalidParameterError);
    const a = makeLowRankMatrix({ randomState: 4 });
    const b = makeLowRankMatrix({ randomState: 4 });
    expect(f64(a)).toEqual(f64(b));
  });
});

describe("makeSPDMatrix", () => {
  it("is exactly symmetric with eigenvalues in [1, 2)", () => {
    for (const nDim of [1, 2, 6, 12]) {
      const M = makeSPDMatrix({ nDim, randomState: nDim });
      const d = f64(M);
      for (let i = 0; i < nDim; i++) {
        for (let j = 0; j < nDim; j++) {
          expect(d[i * nDim + j]).toBe(d[j * nDim + i]);
        }
      }
      for (const lambda of f64(eigvalsh(M))) {
        expect(lambda).toBeGreaterThan(1 - 1e-9);
        expect(lambda).toBeLessThan(2 + 1e-9);
      }
    }
  });
});

describe("makeMoons", () => {
  it("includes both arc endpoints (np.linspace(0, pi, n))", () => {
    const [X, y] = makeMoons({ nSamples: 7, noise: 0, shuffle: false });
    const d = f64(X);
    // First arc: 3 points on the unit upper half circle from (1, 0) to (-1, 0).
    const upper = [
      [1, 0],
      [0, 1],
      [-1, 1.2246467991473532e-16],
    ];
    upper.forEach(([cx, cy], i) => {
      expect(d[i * 2]).toBeCloseTo(cx as number, 12);
      expect(d[i * 2 + 1]).toBeCloseTo(cy as number, 12);
    });
    // Second arc: 4 points of (1 - cos t, 0.5 - sin t), t in {0, pi/3, 2pi/3, pi}.
    const lower = [
      [0, 0.5],
      [0.5, 0.5 - 0.8660254037844386],
      [1.5, 0.5 - 0.8660254037844386],
      [2, 0.5 - 1.2246467991473532e-16],
    ];
    lower.forEach(([cx, cy], i) => {
      expect(d[(3 + i) * 2]).toBeCloseTo(cx as number, 12);
      expect(d[(3 + i) * 2 + 1]).toBeCloseTo(cy as number, 12);
    });
    expect(Array.from(y.data as Int32Array)).toEqual([0, 0, 0, 1, 1, 1, 1]);
  });

  it("handles a single sample per arc without NaN", () => {
    const [X] = makeMoons({ nSamples: 2, noise: 0, shuffle: false });
    expect(f64(X).every(Number.isFinite)).toBe(true);
  });
});

describe("makeGaussianQuantiles", () => {
  it("assigns classes by distance rank, balanced like scikit-learn", () => {
    const [X, y] = makeGaussianQuantiles({ nSamples: 100, nClasses: 3, randomState: 2 });
    const labels = Array.from(y.data as Int32Array);
    const counts = [0, 0, 0];
    for (const l of labels) counts[l] = (counts[l] as number) + 1;
    expect(counts).toEqual([33, 33, 34]);

    const d = f64(X);
    const maxDist = [0, 0, 0];
    const minDist = [Infinity, Infinity, Infinity];
    for (let i = 0; i < 100; i++) {
      const dist = Math.hypot(d[i * 2] as number, d[i * 2 + 1] as number);
      const l = labels[i] as number;
      maxDist[l] = Math.max(maxDist[l] as number, dist);
      minDist[l] = Math.min(minDist[l] as number, dist);
    }
    expect(maxDist[0]).toBeLessThanOrEqual(minDist[1] as number);
    expect(maxDist[1]).toBeLessThanOrEqual(minDist[2] as number);
  });

  it("stays balanced when many distances tie", () => {
    // One feature: |x| ties are impossible, so force ties via nFeatures=1 and few samples.
    const [, y] = makeGaussianQuantiles({
      nSamples: 10,
      nFeatures: 1,
      nClasses: 5,
      randomState: 1,
    });
    const counts = new Array<number>(5).fill(0);
    for (const l of Array.from(y.data as Int32Array)) counts[l] = (counts[l] as number) + 1;
    expect(counts).toEqual([2, 2, 2, 2, 2]);
  });
});

describe("makeCheckerboard", () => {
  it("splits rows and columns into near-equal non-empty blocks", () => {
    const [, rows, cols] = makeCheckerboard({ shape: [7, 5], nClusters: [3, 2] });
    expect(Array.from(rows.data as Int32Array)).toEqual([0, 0, 0, 1, 1, 2, 2]);
    expect(Array.from(cols.data as Int32Array)).toEqual([0, 0, 0, 1, 1]);
    // The old ceil(n / k) split put 5 rows into 4 clusters as 0,0,1,1,2 (cluster 3 empty).
    const [, r2] = makeCheckerboard({ shape: [5, 4], nClusters: [4, 2] });
    expect(new Set(Array.from(r2.data as Int32Array)).size).toBe(4);
  });

  it("rejects more clusters than rows or columns", () => {
    expect(() => makeCheckerboard({ shape: [3, 10], nClusters: [4, 2] })).toThrow(
      InvalidParameterError
    );
    expect(() => makeCheckerboard({ shape: [10, 3], nClusters: [2, 4] })).toThrow(
      InvalidParameterError
    );
  });
});

describe("makeBiclusters shape validation", () => {
  it("requires a pair of positive integers", () => {
    expect(() => makeBiclusters({ shape: [5] as unknown as [number, number] })).toThrow(
      InvalidParameterError
    );
    expect(() => makeBiclusters({ shape: [5, 0] })).toThrow(InvalidParameterError);
  });
});

describe("makeClassification flipY", () => {
  it("validates flipY in [0, 1]", () => {
    expect(() => makeClassification({ flipY: -0.1 })).toThrow(InvalidParameterError);
    expect(() => makeClassification({ flipY: 1.5 })).toThrow(InvalidParameterError);
    expect(() => makeClassification({ flipY: Number.NaN })).toThrow(InvalidParameterError);
    expect(makeClassification({ flipY: 1, nSamples: 10 })[0].shape).toEqual([10, 20]);
  });
});

describe("makeRegression bias / nInformative", () => {
  it("applies the bias and zeroes uninformative coefficients", () => {
    const [X, y] = makeRegression({
      nSamples: 30,
      nFeatures: 5,
      nInformative: 1,
      bias: 7,
      randomState: 9,
    });
    const x = f64(X);
    const yv = f64(y);
    const w = ((yv[0] as number) - 7) / (x[0] as number);
    for (let i = 0; i < 30; i++) {
      expect((yv[i] as number) - 7).toBeCloseTo(w * (x[i * 5] as number), 9);
    }
  });

  it("keeps X identical across nInformative values for a fixed seed", () => {
    const [A] = makeRegression({ nSamples: 10, nFeatures: 4, nInformative: 4, randomState: 1 });
    const [B] = makeRegression({ nSamples: 10, nFeatures: 4, nInformative: 2, randomState: 1 });
    expect(f64(A)).toEqual(f64(B));
  });

  it("validates the new options", () => {
    expect(() => makeRegression({ nFeatures: 3, nInformative: 4 })).toThrow(InvalidParameterError);
    expect(() => makeRegression({ nInformative: 0 })).toThrow(InvalidParameterError);
    expect(() => makeRegression({ bias: Number.NaN })).toThrow(InvalidParameterError);
  });
});

describe("makeBlobs clusterStd", () => {
  it("accepts one standard deviation per center", () => {
    const [X, y] = makeBlobs({
      nSamples: 40,
      centers: [
        [0, 0],
        [100, 100],
      ],
      clusterStd: [1e-6, 5],
      shuffle: false,
      randomState: 3,
    });
    const d = f64(X);
    const labels = Array.from(y.data as Int32Array);
    let spread1 = 0;
    for (let i = 0; i < 40; i++) {
      if (labels[i] === 0) {
        expect(Math.abs(d[i * 2] as number)).toBeLessThan(1e-4);
      } else {
        spread1 = Math.max(spread1, Math.abs((d[i * 2] as number) - 100));
      }
    }
    expect(spread1).toBeGreaterThan(0.1);
  });

  it("rejects a length mismatch and non-positive entries", () => {
    expect(() => makeBlobs({ centers: 3, clusterStd: [1, 2] })).toThrow(InvalidParameterError);
    expect(() => makeBlobs({ centers: 2, clusterStd: [1, 0] })).toThrow(InvalidParameterError);
    expect(() => makeBlobs({ centers: 2, clusterStd: [1, Number.NaN] })).toThrow(
      InvalidParameterError
    );
  });
});

// ─── DataLoader ──────────────────────────────────────────────────────────────

describe("DataLoader length with a sampler", () => {
  const X = tensor([[1], [2], [3], [4], [5]]);

  it("derives the batch count from sampler.length", () => {
    const subset = new DataLoader(X, undefined, {
      batchSize: 2,
      sampler: new SubsetRandomSampler([0, 2, 4], { seed: 1 }),
    });
    expect(subset.length).toBe(2);
    expect([...subset]).toHaveLength(2);

    const weighted = new DataLoader(X, undefined, {
      batchSize: 3,
      sampler: new WeightedRandomSampler([1, 1, 1, 1, 1], { numSamples: 8, seed: 1 }),
    });
    expect(weighted.length).toBe(3);
    expect([...weighted]).toHaveLength(3);

    const dropped = new DataLoader(X, undefined, {
      batchSize: 3,
      dropLast: true,
      sampler: new WeightedRandomSampler([1, 1, 1, 1, 1], { numSamples: 8, seed: 1 }),
    });
    expect(dropped.length).toBe(2);
    expect([...dropped]).toHaveLength(2);
  });
});

describe("DataLoader batch gathering", () => {
  it("copies rows for every numeric dtype and keeps the dtype", () => {
    for (const dtype of ["float32", "float64", "int32", "uint8"] as const) {
      const X = tensor(
        [
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
          [10, 11, 12],
        ],
        { dtype }
      );
      const loader = new DataLoader(X, undefined, {
        batchSize: 2,
        sampler: new SubsetRandomSampler([3, 0, 2, 1], { seed: 0 }),
      });
      const rows = [...loader].flatMap(([b]) => {
        expect(b.dtype).toBe(dtype);
        expect(b.shape[1]).toBe(3);
        return b.toArray() as number[][];
      });
      const order = [...new SubsetRandomSampler([3, 0, 2, 1], { seed: 0 })];
      expect(rows).toEqual(order.map((i) => [3 * i + 1, 3 * i + 2, 3 * i + 3]));
    }
  });

  it("handles int64 and higher-rank tensors", () => {
    const X64 = tensor([[1], [2], [3]], { dtype: "int64" });
    const [b64] = [...new DataLoader(X64, undefined, { batchSize: 2 })][0] as [
      ReturnType<typeof tensor>,
    ];
    expect(b64.dtype).toBe("int64");
    expect(b64.shape).toEqual([2, 1]);

    const X3 = tensor([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ]);
    const [b3] = [...new DataLoader(X3, undefined, { batchSize: 1, shuffle: false })][1] as [
      ReturnType<typeof tensor>,
    ];
    expect(b3.shape).toEqual([1, 2, 2]);
    expect(b3.toArray()).toEqual([
      [
        [5, 6],
        [7, 8],
      ],
    ]);
  });

  it("falls back correctly for non-contiguous views and string tensors", () => {
    const base = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const T = transpose(base); // shape [3, 2], strides [1, 3]
    const batches = [...new DataLoader(T, undefined, { batchSize: 3 })];
    expect(batches[0]?.[0].toArray()).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);

    const S = tensor(["a", "b", "c"]);
    const sb = [...new DataLoader(S, undefined, { batchSize: 2 })];
    expect(sb[0]?.[0].toArray()).toEqual(["a", "b"]);
    expect(sb[1]?.[0].toArray()).toEqual(["c"]);
  });

  it("returns copies, not views of the source data", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const [[b]] = [...new DataLoader(X, undefined, { batchSize: 2 })] as [[typeof X]];
    expect(b.data).not.toBe(X.data);
  });

  it("pairs X and y rows after shuffling", () => {
    const X = tensor([[0], [10], [20], [30], [40], [50]]);
    const y = tensor([0, 1, 2, 3, 4, 5], { dtype: "int32" });
    const loader = new DataLoader(X, y, { batchSize: 4, shuffle: true, seed: 7 });
    for (const [xb, yb] of loader) {
      const xs = (xb.toArray() as number[][]).map((r) => r[0] as number);
      const ys = yb.toArray() as number[];
      expect(xs).toEqual(ys.map((v) => v * 10));
      expect(yb.dtype).toBe("int32");
    }
  });

  it("reports out-of-range sampler indices with IndexError", () => {
    const X = tensor([[1], [2], [3]]);
    const loader = new DataLoader(X, undefined, {
      batchSize: 2,
      sampler: new SequentialSampler(5),
    });
    expect(() => [...loader]).toThrow(IndexError);
  });
});

describe("DataLoader argument validation", () => {
  const X = tensor([[1], [2]]);

  it("rejects a non-tensor X and a plain-array y", () => {
    expect(() => new DataLoader([1, 2, 3] as unknown as typeof X)).toThrow(InvalidParameterError);
    expect(() => new DataLoader(X, [0, 1] as unknown as undefined, { batchSize: 1 })).toThrow(
      /y must be a Tensor/
    );
  });

  it("rejects a non-function collateFn and a malformed sampler", () => {
    expect(() => new DataLoader(X, undefined, { collateFn: 3 as unknown as () => never })).toThrow(
      InvalidParameterError
    );
    expect(
      () =>
        new DataLoader(X, undefined, {
          sampler: {} as unknown as SequentialSampler,
        })
    ).toThrow(InvalidParameterError);
  });

  it("checks boolean options before the sampler/shuffle conflict", () => {
    expect(
      () =>
        new DataLoader(X, undefined, {
          sampler: new SequentialSampler(2),
          shuffle: "yes" as unknown as boolean,
        })
    ).toThrow(/shuffle must be a boolean/);
  });
});

// ─── Image fetchers (stubbed fetch) ──────────────────────────────────────────

function u32be(value: number): number[] {
  return [(value >>> 24) & 0xff, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff];
}

function idxImages(n: number, rows: number, cols: number, magic = 0x803): Uint8Array {
  const header = [...u32be(magic), ...u32be(n), ...u32be(rows), ...u32be(cols)];
  const pixels = Array.from({ length: n * rows * cols }, (_, i) => (i * 17) % 256);
  return new Uint8Array([...header, ...pixels]);
}

function idxLabels(labels: number[], magic = 0x801): Uint8Array {
  return new Uint8Array([...u32be(magic), ...u32be(labels.length), ...labels]);
}

function stubFetch(handler: (url: string) => Response): ReturnType<typeof vi.fn> {
  const mock = vi.fn(async (input: string | URL | Request) => handler(String(input)));
  vi.stubGlobal("fetch", mock);
  return mock;
}

function gz(bytes: Uint8Array): Uint8Array<ArrayBuffer> {
  return new Uint8Array(gzipSync(bytes));
}

describe("fetchMNIST", () => {
  const images = gz(idxImages(3, 2, 2));
  const labels = gz(idxLabels([7, 1, 3]));

  it("validates options before touching the network", async () => {
    const mock = stubFetch(() => new Response("unused"));
    await expect(fetchMNIST({ split: "val" as unknown as "train" })).rejects.toThrow(
      InvalidParameterError
    );
    for (const maxSamples of [0, -1, 1.5, Number.NaN]) {
      await expect(fetchMNIST({ maxSamples })).rejects.toThrow(InvalidParameterError);
    }
    await expect(fetchMNIST({ baseUrl: "" })).rejects.toThrow(InvalidParameterError);
    expect(mock).not.toHaveBeenCalled();
  });

  it("returns float32 pixels and int32 labels, accepting a base URL without trailing slash", async () => {
    const urls: string[] = [];
    stubFetch((url) => {
      urls.push(url);
      return new Response(url.includes("images") ? images : labels);
    });
    const ds = await fetchMNIST({ baseUrl: "https://mirror.test/mnist", maxSamples: 2 });
    expect(urls[0]).toBe("https://mirror.test/mnist/train-images-idx3-ubyte.gz");
    expect(ds.data.dtype).toBe("float32");
    expect(ds.target.dtype).toBe("int32");
    expect(ds.data.shape).toEqual([2, 4]);
    expect(ds.target.toArray()).toEqual([7, 1]);
    expect((ds.data.toArray() as number[][])[0]?.[1]).toBeCloseTo(17 / 255, 6);
  });

  it("rejects wrong magic numbers and truncated files", async () => {
    stubFetch(
      (url) => new Response(url.includes("images") ? gz(idxImages(3, 2, 2, 0x999)) : labels)
    );
    await expect(fetchMNIST({ baseUrl: "https://m.test/" })).rejects.toThrow(/IDX3/);

    stubFetch(
      (url) => new Response(url.includes("images") ? images : gz(idxLabels([1, 2, 3], 0x7)))
    );
    await expect(fetchMNIST({ baseUrl: "https://m.test/" })).rejects.toThrow(/IDX1/);

    const full = idxImages(3, 2, 2);
    stubFetch(
      (url) => new Response(url.includes("images") ? gz(full.subarray(0, full.length - 5)) : labels)
    );
    await expect(fetchMNIST({ baseUrl: "https://m.test/" })).rejects.toThrow(/truncated/);
  });

  it("reports data that is not gzip as a DeepboxError", async () => {
    stubFetch(() => new Response(new Uint8Array([1, 2, 3, 4, 5, 6])));
    await expect(fetchMNIST({ baseUrl: "https://m.test/" })).rejects.toThrow(DeepboxError);
  });
});

function tarOf(files: { name: string; data: Uint8Array }[]): Uint8Array {
  const blocks: Uint8Array[] = [];
  for (const file of files) {
    const header = new Uint8Array(512);
    header.set(new TextEncoder().encode(file.name).subarray(0, 100), 0);
    header.set(new TextEncoder().encode(file.data.length.toString(8).padStart(11, "0")), 124);
    header[156] = 0x30;
    blocks.push(header, file.data, new Uint8Array((512 - (file.data.length % 512)) % 512));
  }
  blocks.push(new Uint8Array(1024));
  const out = new Uint8Array(blocks.reduce((a, b) => a + b.length, 0));
  let off = 0;
  for (const b of blocks) {
    out.set(b, off);
    off += b.length;
  }
  return out;
}

function cifarBatch(labels: number[]): Uint8Array {
  const rec = 1 + 3072;
  const out = new Uint8Array(labels.length * rec);
  labels.forEach((l, i) => {
    out[i * rec] = l;
    out.fill(255, i * rec + 1, i * rec + rec);
  });
  return out;
}

describe("fetchCIFAR10", () => {
  const trainNames = [1, 2, 3, 4, 5].map((i) => `cifar-10-batches-bin/data_batch_${i}.bin`);

  it("fails loudly when a training batch is missing instead of returning a partial set", async () => {
    const archive = gz(
      tarOf([
        ...trainNames.slice(0, 4).map((name) => ({ name, data: cifarBatch([1]) })),
        { name: "cifar-10-batches-bin/test_batch.bin", data: cifarBatch([2]) },
      ])
    );
    stubFetch(() => new Response(archive));
    await expect(fetchCIFAR10({ baseUrl: "https://c.test" })).rejects.toThrow(
      /missing data_batch_5\.bin/
    );
    const test = await fetchCIFAR10({ baseUrl: "https://c.test", split: "test" });
    expect(test.target.toArray()).toEqual([2]);
  });

  it("rejects batches whose size is not a whole number of records", async () => {
    const bad = cifarBatch([1]).subarray(0, 100);
    const archive = gz(tarOf([{ name: "x/test_batch.bin", data: bad }]));
    stubFetch(() => new Response(archive));
    await expect(fetchCIFAR10({ baseUrl: "https://c.test/", split: "test" })).rejects.toThrow(
      /record size/
    );
  });

  it("returns float32 data in [0, 1] and int32 labels, honoring maxSamples", async () => {
    const archive = gz(
      tarOf(trainNames.map((name, i) => ({ name, data: cifarBatch([i, i + 1]) })))
    );
    stubFetch(() => new Response(archive));
    const ds = await fetchCIFAR10({ baseUrl: "https://c.test/", maxSamples: 3 });
    expect(ds.data.shape).toEqual([3, 3072]);
    expect(ds.data.dtype).toBe("float32");
    expect(ds.target.dtype).toBe("int32");
    expect(ds.target.toArray()).toEqual([0, 1, 1]);
    expect((ds.data.toArray() as number[][])[2]?.every((v) => v === 1)).toBe(true);
  });

  it("validates options before fetching", async () => {
    const mock = stubFetch(() => new Response("unused"));
    await expect(fetchCIFAR10({ split: "dev" as unknown as "test" })).rejects.toThrow(
      InvalidParameterError
    );
    await expect(fetchCIFAR10({ maxSamples: 0 })).rejects.toThrow(InvalidParameterError);
    expect(mock).not.toHaveBeenCalled();
  });
});

// ─── Kaggle ──────────────────────────────────────────────────────────────────

describe("Kaggle dataset id validation", () => {
  const credentials = { username: "u", key: "k" };

  it("rejects malformed ids before looking for credentials", async () => {
    vi.stubEnv("KAGGLE_USERNAME", "");
    vi.stubEnv("KAGGLE_KEY", "");
    for (const id of ["nope", "a/b/c", "a/..", "../b", "/b", "a/", "a/ b", "a/b?x=1", "a/b#c"]) {
      await expect(fetchKaggleDatasetInfo(id)).rejects.toThrow(/owner\/dataset/);
    }
    await expect(
      fetchKaggleDataset(undefined as unknown as string, { credentials })
    ).rejects.toThrow(InvalidParameterError);
  });

  it("accepts ordinary slugs", async () => {
    const mock = stubFetch(() => new Response(JSON.stringify({ ref: "o/d-1_x.y" })));
    const info = await fetchKaggleDatasetInfo("o/d-1_x.y", { credentials });
    expect(info.id).toBe("o/d-1_x.y");
    expect(mock).toHaveBeenCalledTimes(1);
  });
});

describe("Kaggle HTTP behavior", () => {
  const credentials = { username: "u", key: "k" };

  it("classifies 4xx as invalid parameters and other failures as DeepboxError", async () => {
    stubFetch(() => new Response("no", { status: 404, statusText: "Not Found" }));
    await expect(fetchKaggleDatasetInfo("a/b", { credentials })).rejects.toThrow(
      InvalidParameterError
    );

    stubFetch(() => new Response("down", { status: 503, statusText: "Service Unavailable" }));
    const err = await fetchKaggleDatasetInfo("a/b", { credentials }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(DeepboxError);
    expect(err).not.toBeInstanceOf(InvalidParameterError);
    expect((err as Error).message).toMatch(/Kaggle API error: 503/);

    stubFetch(() => new Response("slow down", { status: 429, statusText: "Too Many Requests" }));
    const rate = await fetchKaggleDataset("a/b", { credentials }).catch((e: unknown) => e);
    expect(rate).not.toBeInstanceOf(InvalidParameterError);
  });

  it("stops reading at maxBytes and cancels the stream", async () => {
    let cancelled = false;
    let pulls = 0;
    stubFetch(() => {
      const stream = new ReadableStream<Uint8Array>({
        pull(controller) {
          pulls++;
          controller.enqueue(new Uint8Array([pulls, pulls, pulls, pulls]));
        },
        cancel() {
          cancelled = true;
        },
      });
      return new Response(stream);
    });
    const res = await fetchKaggleDataset("a/b", { credentials, maxBytes: 6 });
    expect(Array.from(res.data)).toEqual([1, 1, 1, 1, 2, 2]);
    expect(res.totalBytes).toBe(6);
    expect(cancelled).toBe(true);
    expect(pulls).toBeLessThan(10);
  });

  it("returns the whole body when it is smaller than maxBytes, and handles maxBytes 0", async () => {
    stubFetch(() => new Response(new Uint8Array([9, 8, 7])));
    const small = await fetchKaggleDataset("a/b", { credentials, maxBytes: 100 });
    expect(Array.from(small.data)).toEqual([9, 8, 7]);
    const none = await fetchKaggleDataset("a/b", { credentials, maxBytes: 0 });
    expect(none.data.length).toBe(0);
    expect(none.totalBytes).toBe(0);
  });

  it("rejects a file option that would escape the dataset path", async () => {
    const mock = stubFetch(() => new Response("unused"));
    for (const file of ["..", ".", 5 as unknown as string]) {
      await expect(fetchKaggleDataset("a/b", { credentials, file })).rejects.toThrow(
        InvalidParameterError
      );
    }
    expect(mock).not.toHaveBeenCalled();
  });

  it("validates maxBytes", async () => {
    for (const maxBytes of [-1, 1.5, Number.NaN]) {
      await expect(fetchKaggleDataset("a/b", { credentials, maxBytes })).rejects.toThrow(
        InvalidParameterError
      );
    }
  });

  it("encodes non-ASCII credentials as UTF-8 and trims trailing slashes of apiBaseUrl", async () => {
    let seenUrl = "";
    let seenAuth = "";
    stubFetch(() => new Response("[]"));
    const mock = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      seenUrl = String(input);
      seenAuth = new Headers(init?.headers).get("authorization") ?? "";
      return new Response("[]");
    });
    vi.stubGlobal("fetch", mock);
    await searchKaggleDatasets("iris flowers", {
      credentials: { username: "José", key: "ключ" },
      apiBaseUrl: "https://k.test/api///",
    });
    expect(seenUrl).toBe("https://k.test/api/datasets/list?search=iris%20flowers");
    expect(seenAuth).toBe(`Basic ${Buffer.from("José:ключ", "utf-8").toString("base64")}`);
  });

  it("rejects unexpected response shapes with a DeepboxError", async () => {
    stubFetch(() => new Response(JSON.stringify({ error: "boom" })));
    await expect(searchKaggleDatasets("x", { credentials })).rejects.toThrow(DeepboxError);
    stubFetch(() => new Response("not json"));
    await expect(fetchKaggleDatasetInfo("a/b", { credentials })).rejects.toThrow(/invalid JSON/);
    await expect(searchKaggleDatasets(5 as unknown as string, { credentials })).rejects.toThrow(
      InvalidParameterError
    );
  });

  it("skips malformed entries and replaces non-finite numbers with 0", async () => {
    stubFetch(
      () =>
        new Response(
          JSON.stringify([{ ref: "a/b", totalBytes: "oops", fileCount: 3 }, null, 7, "x"])
        )
    );
    const found = await searchKaggleDatasets("x", { credentials });
    expect(found).toHaveLength(1);
    expect(found[0]?.totalBytes).toBe(0);
    expect(found[0]?.fileCount).toBe(3);
  });
});

describe("readKaggleCredentials", () => {
  it("reads KAGGLE_USERNAME / KAGGLE_KEY first", async () => {
    vi.stubEnv("KAGGLE_USERNAME", "envuser");
    vi.stubEnv("KAGGLE_KEY", "envkey");
    expect(await readKaggleCredentials()).toEqual({ username: "envuser", key: "envkey" });
  });

  it("reads kaggle.json from KAGGLE_CONFIG_DIR", async () => {
    const dir = await mkdtemp(join(tmpdir(), "deepbox-kaggle-"));
    await writeFile(join(dir, "kaggle.json"), JSON.stringify({ username: "fu", key: "fk" }));
    vi.stubEnv("KAGGLE_USERNAME", "");
    vi.stubEnv("KAGGLE_KEY", "");
    vi.stubEnv("KAGGLE_CONFIG_DIR", dir);
    expect(await readKaggleCredentials()).toEqual({ username: "fu", key: "fk" });
  });

  it("returns undefined when nothing is configured", async () => {
    const dir = await mkdtemp(join(tmpdir(), "deepbox-kaggle-empty-"));
    vi.stubEnv("KAGGLE_USERNAME", "");
    vi.stubEnv("KAGGLE_KEY", "");
    vi.stubEnv("KAGGLE_CONFIG_DIR", dir);
    expect(await readKaggleCredentials()).toBeUndefined();
  });
});

// ─── Bundled data tables ─────────────────────────────────────────────────────

describe("bundled dataset tables match scikit-learn", () => {
  const sum = (values: ArrayLike<number>): number => {
    let s = 0;
    for (let i = 0; i < values.length; i++) s += values[i] as number;
    return s;
  };

  it("breast cancer contains no placeholder expressions and matches sklearn", () => {
    expect(BREAST_CANCER_DATA).toHaveLength(BREAST_CANCER_ROWS * BREAST_CANCER_COLS);
    expect(BREAST_CANCER_DATA.every((v) => typeof v === "number" && Number.isFinite(v))).toBe(true);
    // Row 526, column 12 (perimeter error) was committed as Math.LOG2E (1.4427...).
    // biome-ignore lint/suspicious/noApproximativeNumericConstant: 1.443 is the sklearn data value
    expect(BREAST_CANCER_DATA[526 * 30 + 12]).toBe(1.443);
    expect(sum(BREAST_CANCER_DATA)).toBeCloseTo(1056474.4596356, 3);
    expect(sum(BREAST_CANCER_TARGET)).toBe(357);
    expect(loadBreastCancer().data.shape).toEqual([569, 30]);
  });

  it("iris, wine, digits and diabetes match sklearn sums", () => {
    expect(sum(IRIS_DATA)).toBeCloseTo(2078.7, 9);
    expect(sum(IRIS_TARGET)).toBe(150);
    expect(sum(WINE_DATA)).toBeCloseTo(159975.295999, 6);
    expect(sum(WINE_TARGET)).toBe(167);
    expect(sum(decodeDigitsData())).toBe(561718);
    expect(sum(decodeDigitsTarget())).toBe(8070);
    expect(decodeDigitsData()).toHaveLength(DIGITS_ROWS * DIGITS_COLS);
    expect(sum(DIABETES_TARGET)).toBe(67243);
    expect(DIABETES_DATA).toHaveLength(DIABETES_ROWS * DIABETES_COLS);
  });
});
