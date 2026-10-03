import { describe, expect, it } from "vitest";
import {
  catchWarnings,
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../src/core";
import { MeanShift, MiniBatchKMeans, OPTICS, SpectralClustering } from "../../src/ml";
import { toFloat64View } from "../../src/ml/_validation";
import { type Tensor, tensor } from "../../src/ndarray";
import { clearSeed, setSeed } from "../../src/random";

const f64 = (rows: number | number[] | number[][]): Tensor => tensor(rows, { dtype: "float64" });
const vals = (t: Tensor): number[] => Array.from(toFloat64View(t));
const rows = (t: Tensor): number[][] => t.toArray() as number[][];

/** Same partition regardless of how the labels are numbered. */
function samePartition(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  if (a.length !== b.length) return false;
  const ab = new Map<number, number>();
  const ba = new Map<number, number>();
  for (let i = 0; i < a.length; i++) {
    const x = a[i] as number;
    const y = b[i] as number;
    if ((ab.get(x) ?? y) !== y || (ba.get(y) ?? x) !== x) return false;
    ab.set(x, y);
    ba.set(y, x);
  }
  return true;
}

function expectRowsClose(actual: number[][], expected: number[][], digits = 9): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    for (let j = 0; j < (expected[i] as number[]).length; j++) {
      expect((actual[i] as number[])[j]).toBeCloseTo(
        (expected[i] as number[])[j] as number,
        digits
      );
    }
  }
}

// All reference values below come from scikit-learn 1.8 (numpy 2.4, scipy 1.17).

describe("MeanShift (c16)", () => {
  // sklearn: RandomState(0) blobs, rounded to 2 decimals.
  const X = [
    [0.88, 0.2],
    [0.49, 1.12],
    [0.93, -0.49],
    [0.48, -0.08],
    [-0.05, 0.21],
    [0.07, 0.73],
    [4.38, 1.06],
    [4.22, 1.17],
    [4.75, 0.9],
    [4.16, 0.57],
    [2.72, 1.33],
    [1.43, 4.63],
    [2.13, 4.27],
    [1.02, 4.91],
    [1.77, 5.73],
  ];

  it("auto bandwidth matches sklearn estimate_bandwidth (quantile 0.3)", () => {
    // The old estimate skipped the sample itself and used a larger neighbor rank, which gave
    // a different bandwidth and therefore different modes.
    const ms = new MeanShift().fit(f64(X));
    expect(vals(ms.labels)).toEqual([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 2, 2, 2, 4]);
    expectRowsClose(rows(ms.clusterCenters), [
      [0.466666666667, 0.281666666667],
      [4.3775, 0.925],
      [1.526666666667, 4.603333333333],
      [2.72, 1.33],
      [1.77, 5.73],
    ]);
    expect(ms.nIter).toBe(4);
  });

  it("keeps the strongest mode of a group instead of averaging converged seeds", () => {
    const ms = new MeanShift({ bandwidth: 1.2 }).fit(f64(X));
    expect(vals(ms.labels)).toEqual([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 2, 2, 2, 2]);
    expectRowsClose(rows(ms.clusterCenters), [
      [0.466666666667, 0.281666666667],
      [4.3775, 0.925],
      [1.5875, 4.885],
      [2.72, 1.33],
    ]);
  });

  it("binSeeding gives the same modes as sklearn", () => {
    const ms = new MeanShift({ bandwidth: 1.2, binSeeding: true }).fit(f64(X));
    expect(vals(ms.labels)).toEqual([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 2, 2, 2, 2]);
    expectRowsClose(rows(ms.clusterCenters), [
      [0.466666666667, 0.281666666667],
      [4.3775, 0.925],
      [1.5875, 4.885],
      [2.72, 1.33],
    ]);
  });

  it("clusterAll: false labels samples farther than the bandwidth from every mode as -1", () => {
    const ms = new MeanShift({ bandwidth: 0.8, clusterAll: false }).fit(f64(X));
    expect(vals(ms.labels)).toEqual([1, 3, 1, 1, 3, 3, 0, 0, 0, 0, 4, 2, 2, 2, 5]);
    const all = new MeanShift({ bandwidth: 0.8 }).fit(f64(X));
    expect(vals(all.labels).every((l) => l >= 0)).toBe(true);
  });

  it("drops bin seeds that no sample supports instead of returning them as centers", () => {
    // In 5 dimensions the bin center of [0.49, ...] is sqrt(5) * 0.49 = 1.1 away from the sample.
    const data = f64([
      [0.49, 0.49, 0.49, 0.49, 0.49],
      [10, 10, 10, 10, 10],
      [10.2, 10.2, 10.2, 10.2, 10.2],
    ]);
    const ms = new MeanShift({ bandwidth: 1, binSeeding: true }).fit(data);
    expect(ms.clusterCenters.shape).toEqual([1, 5]);
    expect(vals(ms.clusterCenters)[0]).toBeCloseTo(10.1, 6);
    expect(vals(ms.labels)).toEqual([0, 0, 0]);

    // No seed left at all: a clear error instead of an empty model.
    const lonely = f64([[0.49, 0.49, 0.49, 0.49, 0.49]]);
    expect(() =>
      new MeanShift({ bandwidth: 1, binSeeding: true, minBinFreq: 2 }).fit(lonely)
    ).toThrow(InvalidParameterError);
  });

  it("tol is relative to the bandwidth, so the result does not depend on the data scale", () => {
    const scaled = X.map((r) => r.map((v) => v * 1e-4));
    const a = new MeanShift({ bandwidth: 1.2 }).fit(f64(X));
    const b = new MeanShift({ bandwidth: 1.2e-4 }).fit(f64(scaled));
    expect(vals(b.labels)).toEqual(vals(a.labels));
    expect(b.nIter).toBe(a.nIter);
  });

  it("rejects NaN, zero and negative bandwidths and a bad tol", () => {
    expect(() => new MeanShift({ bandwidth: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new MeanShift({ bandwidth: 0 })).toThrow(InvalidParameterError);
    expect(() => new MeanShift({ tol: -1 })).toThrow(InvalidParameterError);
    expect(() => new MeanShift({ tol: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new MeanShift({ minBinFreq: 0 })).toThrow(InvalidParameterError);
    const ms = new MeanShift();
    expect(() => ms.setParams({ bandwidth: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => ms.setParams({ tol: Number.POSITIVE_INFINITY })).toThrow(InvalidParameterError);
    expect(() => ms.setParams({ clusterAll: "yes" })).toThrow(InvalidParameterError);
  });

  it("throws a typed error when the bandwidth cannot be estimated", () => {
    expect(() => new MeanShift().fit(f64([[1, 1]]))).toThrow(InvalidParameterError);
    expect(() =>
      new MeanShift().fit(
        f64([
          [2, 2],
          [2, 2],
          [2, 2],
        ])
      )
    ).toThrow(InvalidParameterError);
  });

  it("auto bandwidth needs at least 4 samples and an infinite bandwidth works with binSeeding", () => {
    // floor(0.3 * n) is 0 below 4 samples, so the estimate is the distance to the sample itself.
    expect(() => new MeanShift().fit(f64([[0], [1], [5]]))).toThrow(InvalidParameterError);
    const ms = new MeanShift({ bandwidth: Number.POSITIVE_INFINITY, binSeeding: true }).fit(
      f64([[1], [2], [3]])
    );
    expect(vals(ms.labels)).toEqual([0, 0, 0]);
    expect(vals(ms.clusterCenters)[0]).toBeCloseTo(2, 12);
  });

  it("works for integer and float32 input and does not modify X", () => {
    const xi = tensor(
      [
        [0, 0],
        [1, 0],
        [10, 10],
        [11, 10],
      ],
      { dtype: "int32" }
    );
    const before = vals(xi);
    const ms = new MeanShift({ bandwidth: 3 }).fit(xi);
    // Equal intensity: the mode with the larger coordinates comes first, as in sklearn.
    expect(vals(ms.labels)).toEqual([1, 1, 0, 0]);
    expect(vals(xi)).toEqual(before);
    expect(vals(ms.predict(tensor([[0.2, 0.1]], { dtype: "float32" })))).toHaveLength(1);
  });

  it("refit replaces the model and predict checks the feature count", () => {
    const ms = new MeanShift({ bandwidth: 1.2 });
    expect(() => ms.predict(f64([[0, 0]]))).toThrow(NotFittedError);
    ms.fit(f64(X));
    ms.fit(f64([[0], [0.1], [5], [5.1]]));
    expect(ms.clusterCenters.shape).toEqual([2, 1]);
    expect(() => ms.predict(f64([[0, 0]]))).toThrow(ShapeError);
    expect(vals(ms.predict(f64([[0.2], [4.9]])))).toEqual([vals(ms.labels)[0], vals(ms.labels)[2]]);
  });
});

describe("MiniBatchKMeans (c16)", () => {
  const blobs = (): number[][] => {
    const out: number[][] = [];
    for (let i = 0; i < 120; i++) {
      const c = i % 3;
      out.push([c * 10 + (i % 7) * 0.05, c * -8 + ((i * 3) % 7) * 0.05]);
    }
    return out;
  };

  it("accepts negative, fractional and huge seeds (the old generator indexed rows with negative numbers)", () => {
    for (const seed of [-1, -100000, 3.7, 1e12, 0]) {
      const km = new MiniBatchKMeans({ nClusters: 3, batchSize: 16, randomState: seed });
      km.fit(f64(blobs()));
      expect(km.inertia).toBeLessThan(10);
    }
  });

  it("is reproducible for a seed, differs between seeds and follows setSeed without one", () => {
    const X = f64(blobs());
    const a = new MiniBatchKMeans({ nClusters: 3, batchSize: 8, randomState: 5 }).fit(X);
    const b = new MiniBatchKMeans({ nClusters: 3, batchSize: 8, randomState: 5 }).fit(X);
    expect(vals(a.clusterCenters)).toEqual(vals(b.clusterCenters));
    const c = new MiniBatchKMeans({ nClusters: 3, batchSize: 8, randomState: 6 }).fit(X);
    expect(vals(c.clusterCenters)).not.toEqual(vals(a.clusterCenters));

    setSeed(11);
    const d = new MiniBatchKMeans({ nClusters: 3, batchSize: 8 }).fit(X);
    setSeed(11);
    const e = new MiniBatchKMeans({ nClusters: 3, batchSize: 8 }).fit(X);
    clearSeed();
    expect(vals(d.clusterCenters)).toEqual(vals(e.clusterCenters));
  });

  it("finds well separated blobs and reports a consistent inertia", () => {
    const X = f64(blobs());
    const km = new MiniBatchKMeans({ nClusters: 3, batchSize: 32, randomState: 1 }).fit(X);
    const labels = vals(km.labels);
    for (let i = 0; i < 120; i++) expect(labels[i]).toBe(labels[i % 3]);
    expect(new Set(labels).size).toBe(3);

    const centers = rows(km.clusterCenters);
    const data = blobs();
    let inertia = 0;
    data.forEach((p, i) => {
      const c = centers[labels[i] as number] as number[];
      inertia +=
        ((p[0] as number) - (c[0] as number)) ** 2 + ((p[1] as number) - (c[1] as number)) ** 2;
    });
    expect(km.inertia).toBeCloseTo(inertia, 8);
    expect(km.score(X)).toBeCloseTo(-inertia, 8);
    expect(vals(km.predict(X))).toEqual(labels);
  });

  it("predict uses the fitted centers even after nClusters is changed with setParams", () => {
    const X = f64(blobs());
    const km = new MiniBatchKMeans({ nClusters: 3, batchSize: 16, randomState: 2 }).fit(X);
    const before = vals(km.predict(X));
    km.setParams({ nClusters: 6 });
    expect(vals(km.predict(X))).toEqual(before);
    expect(km.transform(X).shape).toEqual([120, 3]);
  });

  it("validates every option in the constructor and in setParams", () => {
    expect(() => new MiniBatchKMeans({ maxIter: 0 })).toThrow(InvalidParameterError);
    expect(() => new MiniBatchKMeans({ maxIter: 1.5 })).toThrow(InvalidParameterError);
    expect(() => new MiniBatchKMeans({ nInit: 0 })).toThrow(InvalidParameterError);
    expect(() => new MiniBatchKMeans({ tol: -1 })).toThrow(InvalidParameterError);
    expect(() => new MiniBatchKMeans({ tol: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new MiniBatchKMeans({ init: "bad" as "random" })).toThrow(InvalidParameterError);
    expect(() => new MiniBatchKMeans({ randomState: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new MiniBatchKMeans({ maxNoImprovement: 0 })).toThrow(InvalidParameterError);
    const km = new MiniBatchKMeans();
    expect(() => km.setParams({ maxNoImprovement: -2 })).toThrow(InvalidParameterError);
    expect(() => km.setParams({ tol: Number.POSITIVE_INFINITY })).toThrow(InvalidParameterError);
    expect(km.setParams({ maxNoImprovement: Number.POSITIVE_INFINITY })).toBe(km);
    expect(km.getParams().maxNoImprovement).toBe(Number.POSITIVE_INFINITY);
  });

  it("tol stops training early and is relative to the data scale", () => {
    const a = new MiniBatchKMeans({
      nClusters: 3,
      batchSize: 16,
      tol: 1e-3,
      maxIter: 500,
      maxNoImprovement: Number.POSITIVE_INFINITY,
      nInit: 1,
      randomState: 3,
    }).fit(f64(blobs()));
    expect(a.nSteps).toBeLessThan(500);
    const scaled = blobs().map((r) => r.map((v) => v * 1000));
    const b = new MiniBatchKMeans({
      nClusters: 3,
      batchSize: 16,
      tol: 1e-3,
      maxIter: 500,
      maxNoImprovement: Number.POSITIVE_INFINITY,
      nInit: 1,
      randomState: 3,
    }).fit(f64(scaled));
    expect(b.nSteps).toBe(a.nSteps);
    expect(vals(b.labels)).toEqual(vals(a.labels));
  });

  it("maxNoImprovement ends training before maxIter on converged data", () => {
    const km = new MiniBatchKMeans({
      nClusters: 3,
      batchSize: 16,
      maxIter: 1000,
      maxNoImprovement: 5,
      nInit: 1,
      randomState: 3,
    }).fit(f64(blobs()));
    expect(km.nSteps).toBeLessThan(1000);
  });

  it("handles fewer distinct samples than clusters without failing", () => {
    const X = f64([
      [1, 1],
      [1, 1],
      [1, 1],
      [2, 2],
      [2, 2],
    ]);
    for (const init of ["kmeans++", "random"] as const) {
      const warnings = catchWarnings(() => {
        const km = new MiniBatchKMeans({ nClusters: 4, batchSize: 5, init, randomState: 0 }).fit(X);
        expect(km.clusterCenters.shape).toEqual([4, 2]);
        for (const l of vals(km.labels)) expect(l).toBeGreaterThanOrEqual(0);
      });
      expect(warnings.some((w) => w.category === "ConvergenceWarning")).toBe(true);
    }
  });

  it("random init picks distinct samples even when nClusters equals n_samples", () => {
    const X = f64([[0], [1], [2], [3]]);
    const km = new MiniBatchKMeans({ nClusters: 4, init: "random", randomState: 9, nInit: 1 }).fit(
      X
    );
    expect(vals(km.clusterCenters).sort()).toEqual([0, 1, 2, 3]);
    expect(km.inertia).toBe(0);
  });

  it("partialFit keeps each center equal to the running mean of its samples", () => {
    const km = new MiniBatchKMeans({ nClusters: 1, randomState: 0 });
    km.partialFit(f64([[0], [2]]));
    expect(vals(km.clusterCenters)).toEqual([1]);
    km.partialFit(f64([[4]]));
    expect(vals(km.clusterCenters)[0]).toBeCloseTo(2, 12);
    km.partialFit(f64([[6], [8], [10]]));
    expect(vals(km.clusterCenters)[0]).toBeCloseTo(5, 12);
    expect(km.nSteps).toBe(3);
    expect(vals(km.labels)).toEqual([0, 0, 0]);
    expect(km.inertia).toBeCloseTo((6 - 5) ** 2 + (8 - 5) ** 2 + (10 - 5) ** 2, 10);
  });

  it("partialFit assigns the whole batch with the centers from before the step", () => {
    const km = new MiniBatchKMeans({ nClusters: 2, init: "random", randomState: 0 });
    km.partialFit(f64([[0], [10]]));
    expect(vals(km.clusterCenters).sort((a, b) => a - b)).toEqual([0, 10]);
    km.partialFit(f64([[1], [9], [3]]));
    // Each of the two centers has seen one sample; 1 and 3 join the center at 0, 9 the center at 10.
    expect(vals(km.clusterCenters).sort((a, b) => a - b)[0]).toBeCloseTo(
      0 + (1 - 0) / 2 + (3 - 0.5) / 3,
      12
    );
    expect(vals(km.clusterCenters).sort((a, b) => a - b)[1]).toBeCloseTo(9.5, 12);
  });

  it("partialFit validates its input", () => {
    const km = new MiniBatchKMeans({ nClusters: 3 });
    expect(() => km.partialFit(f64([[0], [1]]))).toThrow(InvalidParameterError);
    km.partialFit(
      f64([
        [0, 0],
        [1, 1],
        [5, 5],
      ])
    );
    expect(() => km.partialFit(f64([[0]]))).toThrow(ShapeError);
    expect(() => km.partialFit(f64([[Number.NaN, 0]]))).toThrow(DataValidationError);
    // A later fit starts from scratch.
    km.fit(
      f64([
        [0, 0],
        [1, 1],
        [5, 5],
        [6, 6],
      ])
    );
    expect(km.clusterCenters.shape).toEqual([3, 2]);
  });

  it("transform, fitTransform and score agree with predict", () => {
    const X = f64(blobs());
    const km = new MiniBatchKMeans({ nClusters: 3, batchSize: 16, randomState: 4 });
    const d = rows(km.fitTransform(X));
    const labels = vals(km.predict(X));
    d.forEach((r, i) => {
      expect(r.indexOf(Math.min(...r))).toBe(labels[i]);
    });
    expect(() => km.transform(f64([[1, 2, 3]]))).toThrow(ShapeError);
    expect(() => new MiniBatchKMeans().score(X)).toThrow(NotFittedError);
  });

  it("does not modify X and supports integer input", () => {
    const xi = tensor([[0], [1], [100], [101], [200], [201]], { dtype: "int32" });
    const before = vals(xi);
    const km = new MiniBatchKMeans({ nClusters: 3, batchSize: 6, randomState: 1 }).fit(xi);
    expect(vals(xi)).toEqual(before);
    expect(new Set(vals(km.labels)).size).toBe(3);
  });
});

describe("OPTICS (c16)", () => {
  const X = [
    [0, 0],
    [0.2, 0.1],
    [0.1, 0.35],
    [0.3, 0.2],
    [0.25, 0],
    [5, 5],
    [5.1, 5.2],
    [5.3, 5.1],
    [4.9, 5],
    [5.2, 4.8],
    [2.5, 2.5],
    [9, 0],
  ];
  const INF = Number.POSITIVE_INFINITY;
  const refReach = [
    INF,
    0.25,
    0.25,
    0.141421356237,
    0.141421356237,
    0.282842712475,
    0.22360679775,
    0.22360679775,
    3.465544690233,
    0.282842712475,
    3.182766092568,
    6.122091146006,
  ];
  const refCore = [
    0.25, 0.141421356237, 0.269258240357, 0.206155281281, 0.206155281281, 0.22360679775,
    0.22360679775, 0.316227766017, 0.282842712475, 0.316227766017, 3.222188697144, 6.300793600809,
  ];
  const refOrder = [0, 1, 3, 4, 2, 10, 8, 5, 6, 7, 9, 11];
  const refPred = [-1, 0, 3, 1, 1, 8, 5, 6, 10, 5, 3, 9];

  it("xi extraction, ordering, reachability and predecessors match sklearn", () => {
    const o = new OPTICS({ minSamples: 3, clusterMethod: "xi", xi: 0.1 }).fit(f64(X));
    expect(Array.from(o.ordering)).toEqual(refOrder);
    expect(Array.from(o.predecessor)).toEqual(refPred);
    refReach.forEach((r, i) => {
      if (r === INF) expect(o.reachability[i]).toBe(INF);
      else expect(o.reachability[i]).toBeCloseTo(r, 9);
    });
    refCore.forEach((c, i) => {
      expect(o.coreDistances[i]).toBeCloseTo(c, 9);
    });
    expect(vals(o.labels)).toEqual([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1, 1]);
    expect(o.clusterHierarchy).toEqual([
      [0, 4],
      [6, 11],
      [0, 11],
    ]);
  });

  it("dbscan extraction matches sklearn and has no hierarchy", () => {
    const o = new OPTICS({ minSamples: 3, clusterMethod: "dbscan", eps: 0.5 }).fit(f64(X));
    expect(vals(o.labels)).toEqual([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1, -1]);
    expect(o.clusterHierarchy).toEqual([]);
  });

  it("breaks reachability ties by the lowest index like sklearn", () => {
    // 3 x 3 grid: many equal distances. The old Map-based queue broke ties by insertion order, and
    // its simplified xi step labeled the whole grid as noise (sklearn: one cluster).
    const grid = f64([
      [0, 0],
      [0, 1],
      [0, 2],
      [1, 0],
      [1, 1],
      [1, 2],
      [2, 0],
      [2, 1],
      [2, 2],
    ]);
    const o = new OPTICS({ minSamples: 3, clusterMethod: "xi" }).fit(grid);
    expect(Array.from(o.ordering)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    expect(Array.from(o.predecessor)).toEqual([-1, 0, 1, 0, 1, 2, 3, 4, 5]);
    expect(vals(o.labels)).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect(o.clusterHierarchy).toEqual([[0, 8]]);
  });

  it("minSamples = 1 makes every point a core point with core distance 0", () => {
    // The old code read dists[-1] and returned Infinity, so everything was noise.
    const o = new OPTICS({ minSamples: 1, eps: 1 }).fit(f64(X));
    expect(Array.from(o.coreDistances)).toEqual(new Array(12).fill(0));
    expect(samePartition(vals(o.labels), [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 3])).toBe(true);
  });

  it("rejects minSamples above n_samples and eps above maxEps", () => {
    expect(() => new OPTICS({ minSamples: 13 }).fit(f64(X))).toThrow(InvalidParameterError);
    expect(() => new OPTICS({ minSamples: 3, maxEps: 1, eps: 2 }).fit(f64(X))).toThrow(
      InvalidParameterError
    );
    // eps only matters for the dbscan cut
    expect(() =>
      new OPTICS({ minSamples: 3, maxEps: 1, eps: 2, clusterMethod: "xi" }).fit(f64(X))
    ).not.toThrow();
  });

  it("validates options in the constructor", () => {
    expect(() => new OPTICS({ maxEps: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new OPTICS({ xi: 1 })).toThrow(InvalidParameterError);
    expect(() => new OPTICS({ xi: 0 })).toThrow(InvalidParameterError);
    expect(() => new OPTICS({ eps: -1 })).toThrow(InvalidParameterError);
    expect(() => new OPTICS({ clusterMethod: "bad" as "xi" })).toThrow(InvalidParameterError);
    expect(() => new OPTICS({ minClusterSize: 1 })).toThrow(InvalidParameterError);
    const o = new OPTICS();
    expect(() => o.setParams({ predecessorCorrection: 1 })).toThrow(InvalidParameterError);
    expect(o.setParams({ minClusterSize: 4, predecessorCorrection: false })).toBe(o);
    expect(o.getParams().minClusterSize).toBe(4);
    expect(o.getParams().predecessorCorrection).toBe(false);
  });

  it("predict validates the input and cluster centers keep a 2-D shape without clusters", () => {
    const o = new OPTICS({ minSamples: 3, eps: 0.5 }).fit(f64(X));
    expect(() => o.predict(f64([[1, 2, 3]]))).toThrow(ShapeError);
    expect(() => o.predict(f64([[Number.NaN, 0]]))).toThrow(DataValidationError);
    expect(
      vals(
        o.predict(
          f64([
            [0.1, 0.1],
            [5.1, 5.1],
            [9.5, 0.5],
          ])
        )
      )
    ).toEqual([0, 1, -1]);

    const none = new OPTICS({ minSamples: 3, eps: 0.01 }).fit(f64(X));
    expect(vals(none.labels).every((l) => l === -1)).toBe(true);
    expect(none.clusterCenters.shape).toEqual([0, 2]);

    const centers = o.clusterCenters;
    expect(centers.shape).toEqual([2, 2]);
    expectRowsClose(rows(centers), [
      [0.17, 0.13],
      [5.1, 5.02],
    ]);
  });

  it("keeps its own copy of the training data", () => {
    const data = f64(X);
    const o = new OPTICS({ minSamples: 3, eps: 0.5 }).fit(data);
    const before = vals(
      o.predict(
        f64([
          [0.1, 0.1],
          [5.1, 5.1],
        ])
      )
    );
    (data.data as Float64Array).fill(1000);
    expect(
      vals(
        o.predict(
          f64([
            [0.1, 0.1],
            [5.1, 5.1],
          ])
        )
      )
    ).toEqual(before);
  });

  it("refit resets every fitted attribute", () => {
    const o = new OPTICS({ minSamples: 3, clusterMethod: "xi", xi: 0.1 }).fit(f64(X));
    expect(o.clusterHierarchy.length).toBe(3);
    o.setParams({ clusterMethod: "dbscan", eps: 0.5 });
    o.fit(f64(X));
    expect(o.clusterHierarchy).toEqual([]);
    expect(() => new OPTICS().clusterHierarchy).toThrow(NotFittedError);
    expect(() => new OPTICS().predecessor).toThrow(NotFittedError);
  });

  it("handles 1500 samples without building an n x n distance matrix", () => {
    const big: number[][] = [];
    for (let i = 0; i < 1500; i++)
      big.push([(i % 30) * 0.1 + (i % 2) * 20, Math.floor(i / 30) * 0.1]);
    const o = new OPTICS({ minSamples: 5, maxEps: 1 }).fit(f64(big));
    expect(o.labels.size).toBe(1500);
    expect(new Set(vals(o.labels)).size).toBeGreaterThanOrEqual(2);
  });
});

describe("SpectralClustering (c16)", () => {
  // sklearn make_moons(40, noise=0.05, random_state=0), rounded to 4 decimals.
  const moons = [
    [0.0339, 0.1886],
    [-0.0678, 0.4943],
    [0.2805, -0.2004],
    [0.7692, 0.5729],
    [-0.1034, 0.9704],
    [-0.7485, 0.6028],
    [0.8626, -0.5172],
    [-0.2421, 0.9797],
    [-1.0092, 0.1116],
    [0.9482, -0.4251],
    [0.5877, -0.4198],
    [1.8094, -0.1083],
    [2.0085, 0.2463],
    [0.9086, 0.1988],
    [1.9246, 0.5575],
    [1.0229, -0.4395],
    [1.7528, -0.1823],
    [-0.9138, 0.4767],
    [1.8607, 0.0221],
    [0.2639, 0.9672],
    [-0.5621, 0.7259],
    [0.7135, 0.7537],
    [-0.892, 0.3343],
    [-0.6346, 0.7366],
    [1.5684, -0.2874],
    [0.3771, 0.9514],
    [1.4574, -0.5235],
    [0.9656, 0.111],
    [0.4641, -0.3933],
    [-0.4542, 0.9664],
    [1.3227, -0.4895],
    [0.2541, -0.1256],
    [0.5875, 0.8516],
    [0.0936, 0.1195],
    [-0.0102, 0.4074],
    [-1.0575, 0.0403],
    [0.1704, 1.0452],
    [1.9088, 0.345],
    [0.9597, 0.4049],
    [0.8778, -0.1028],
  ];
  const ref = [
    1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
    1, 1, 1, 1, 1, 0, 0, 0,
  ];

  it("nearest_neighbors affinity matches sklearn (nNeighbors counts the sample itself)", () => {
    // The old graph linked each sample to nNeighbors other samples with weight 1, which merged
    // the two moons.
    const sc = new SpectralClustering({
      nClusters: 2,
      affinity: "nearest_neighbors",
      nNeighbors: 8,
      randomState: 0,
    }).fit(f64(moons));
    expect(samePartition(vals(sc.labels), ref)).toBe(true);
  });

  it("rbf affinity separates the moons like sklearn", () => {
    const sc = new SpectralClustering({ nClusters: 2, gamma: 10, randomState: 0 }).fit(f64(moons));
    expect(samePartition(vals(sc.labels), ref)).toBe(true);
  });

  it("precomputed affinity gives the same partition as the rbf kernel it was built from", () => {
    const n = moons.length;
    const A: number[][] = moons.map((p) =>
      moons.map((q) =>
        Math.exp(
          -10 *
            (((p[0] as number) - (q[0] as number)) ** 2 +
              ((p[1] as number) - (q[1] as number)) ** 2)
        )
      )
    );
    const sc = new SpectralClustering({
      nClusters: 2,
      affinity: "precomputed",
      randomState: 0,
    }).fit(f64(A));
    expect(samePartition(vals(sc.labels), ref)).toBe(true);
    expect(sc.labels.size).toBe(n);
    // New samples are given as their affinities to the training samples.
    const row = A[0] as number[];
    expect(vals(sc.predict(f64([row])))).toEqual([vals(sc.labels)[0]]);
    expect(() => sc.clusterCenters).toThrow(InvalidParameterError);
    expect(() => sc.predict(f64([[1, 2]]))).toThrow(ShapeError);
  });

  it("precomputed affinity must be square and non-negative", () => {
    const sc = new SpectralClustering({ nClusters: 2, affinity: "precomputed" });
    expect(() =>
      sc.fit(
        f64([
          [1, 0.5, 0.1],
          [0.5, 1, 0.1],
        ])
      )
    ).toThrow(ShapeError);
    expect(() =>
      sc.fit(
        f64([
          [1, -0.5],
          [-0.5, 1],
        ])
      )
    ).toThrow(DataValidationError);
  });

  it("warns when the affinity graph is disconnected", () => {
    const X = f64([[0], [0.1], [100], [100.1]]);
    const warnings = catchWarnings(() => {
      new SpectralClustering({ nClusters: 2, gamma: 1, randomState: 0 }).fit(X);
    });
    expect(warnings.some((w) => w.message.includes("not fully connected"))).toBe(true);
  });

  it("nComponents controls the embedding size and is validated", () => {
    const sc = new SpectralClustering({ nClusters: 2, nComponents: 3, gamma: 10, randomState: 0 });
    sc.fit(f64(moons));
    // sklearn n_components=3 on the same data; with 2 components the moons separate (see above).
    const ref3 = "1111111111100101010111110101110111111011".split("").map(Number);
    expect(samePartition(vals(sc.labels), ref3)).toBe(true);
    expect(samePartition(vals(sc.labels), ref)).toBe(false);
    expect(() => new SpectralClustering({ nComponents: 0 })).toThrow(InvalidParameterError);
    expect(() =>
      new SpectralClustering({ nClusters: 2, nComponents: 100, randomState: 0 }).fit(f64(moons))
    ).toThrow(InvalidParameterError);
    expect(sc.getParams().nComponents).toBe(3);
  });

  it("rejects non-finite gamma and bad options in the constructor", () => {
    expect(() => new SpectralClustering({ gamma: Number.POSITIVE_INFINITY })).toThrow(
      InvalidParameterError
    );
    expect(() => new SpectralClustering({ gamma: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new SpectralClustering({ nNeighbors: 0 })).toThrow(InvalidParameterError);
    expect(() => new SpectralClustering({ nInit: 0 })).toThrow(InvalidParameterError);
    expect(() => new SpectralClustering({ randomState: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => new SpectralClustering({ affinity: "bad" as "rbf" })).toThrow(
      InvalidParameterError
    );
    const sc = new SpectralClustering();
    expect(() => sc.setParams({ gamma: Number.POSITIVE_INFINITY })).toThrow(InvalidParameterError);
    expect(sc.setParams({ affinity: "precomputed" })).toBe(sc);
  });

  it("predict validates the input, centers are per-cluster means and X is not modified", () => {
    const data = f64(moons);
    const before = vals(data);
    const sc = new SpectralClustering({ nClusters: 2, gamma: 10, randomState: 0 }).fit(data);
    expect(vals(data)).toEqual(before);
    expect(() => sc.predict(f64([[1, 2, 3]]))).toThrow(ShapeError);
    expect(() => sc.predict(f64([[Number.NaN, 1]]))).toThrow(DataValidationError);
    expect(vals(sc.predict(f64(moons)))).toEqual(vals(sc.labels));

    const labels = vals(sc.labels);
    const centers = rows(sc.clusterCenters);
    for (const k of [0, 1]) {
      const members = moons.filter((_, i) => labels[i] === k);
      const mx = members.reduce((s, p) => s + (p[0] as number), 0) / members.length;
      const my = members.reduce((s, p) => s + (p[1] as number), 0) / members.length;
      expect((centers[k] as number[])[0]).toBeCloseTo(mx, 10);
      expect((centers[k] as number[])[1]).toBeCloseTo(my, 10);
    }

    // The fitted data is copied: later edits of X do not change predictions.
    const probe = f64([
      [0.1, 0.2],
      [1.9, 0.1],
    ]);
    const pred = vals(sc.predict(probe));
    (data.data as Float64Array).fill(500);
    expect(vals(sc.predict(probe))).toEqual(pred);
  });

  it("single cluster and refit work", () => {
    const sc = new SpectralClustering({ nClusters: 1, gamma: 1, randomState: 0 });
    sc.fit(f64([[0], [0.1], [0.2], [0.3]]));
    expect(vals(sc.labels)).toEqual([0, 0, 0, 0]);
    sc.setParams({ nClusters: 2 });
    sc.fit(f64([[0], [0.1], [5], [5.1]]));
    expect(samePartition(vals(sc.labels), [0, 0, 1, 1])).toBe(true);
  });
});
