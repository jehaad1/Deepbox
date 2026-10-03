import { describe, expect, it } from "vitest";
import {
  ConvergenceError,
  catchWarnings,
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../src/core";
import {
  AffinityPropagation,
  AgglomerativeClustering,
  Birch,
  DBSCAN,
  GaussianMixture,
  getEstimatorTags,
  KMeans,
} from "../../src/ml";
import { type Tensor, tensor } from "../../src/ndarray";

// Reference values below were computed with scikit-learn 1.8 / SciPy 1.17 (numpy 2.4).

/** Flat copy of a tensor's elements as plain numbers. */
function flat(t: Tensor): number[] {
  const out: number[] = [];
  const walk = (v: unknown): void => {
    if (Array.isArray(v)) for (const x of v) walk(x);
    else out.push(Number(v));
  };
  walk(t.toArray());
  return out;
}

/** Relabel in order of first appearance so partitions compare independent of label numbering. */
function canon(labels: readonly number[]): number[] {
  const seen = new Map<number, number>();
  return labels.map((l) => {
    if (!seen.has(l)) seen.set(l, seen.size);
    return seen.get(l) as number;
  });
}

function expectClose(actual: readonly number[], expected: readonly number[], tol: number): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs((actual[i] as number) - (expected[i] as number))).toBeLessThanOrEqual(tol);
  }
}

// ---------------------------------------------------------------------------------------------
// KMeans
// ---------------------------------------------------------------------------------------------

const KM_X = [
  [0.68, -0.19],
  [0.01, 0.16],
  [-0.32, 0.0],
  [-0.0, -0.7],
  [0.41, 0.24],
  [5.75, 0.93],
  [6.2, 0.9],
  [5.9, 0.42],
  [6.22, 1.05],
  [6.11, 0.39],
  [2.66, 7.06],
  [1.85, 7.81],
  [1.98, 6.42],
  [1.84, 6.08],
  [2.42, 6.83],
];

describe("KMeans", () => {
  const X = tensor(KM_X, { dtype: "float64" });

  for (const algorithm of ["lloyd", "elkan"] as const) {
    it(`matches scikit-learn inertia, centers, transform and score (${algorithm})`, () => {
      const km = new KMeans({ nClusters: 3, randomState: 0, algorithm }).fit(X);
      expect(km.inertia).toBeCloseTo(4.013199999999999, 10);

      const centers = km.clusterCenters.toArray() as number[][];
      const order = [0, 1, 2].sort((a, b) => (centers[a]?.[0] ?? 0) - (centers[b]?.[0] ?? 0));
      const sorted = order.map((i) => centers[i] as number[]);
      expectClose(sorted.flat(), [0.156, -0.098, 2.15, 6.84, 6.036, 0.738], 1e-12);

      const labels = flat(km.labels);
      expect(canon(labels)).toEqual([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]);

      const dist = flat(km.transform(tensor(KM_X.slice(0, 2), { dtype: "float64" })));
      const expected = [
        [0.532015037381, 7.182047061946, 5.435799849148],
        [0.296445610526, 7.014413731738, 6.053656746133],
      ];
      const reordered = [0, 1].flatMap((r) => order.map((c) => dist[r * 3 + c] as number));
      expectClose(reordered, expected.flat(), 1e-10);

      expect(km.score(tensor(KM_X.slice(0, 4), { dtype: "float64" }))).toBeCloseTo(
        -0.9938399999999998,
        10
      );
    });
  }

  it("keeps centroids and labels in full float64 precision", () => {
    // float32 centers used to give 0.20000000298 for the mean of 0.1 and 0.3.
    const Y = tensor(
      [
        [0.1, 0.2],
        [0.3, 0.7],
        [10.1, 10.2],
        [10.3, 10.7],
      ],
      { dtype: "float64" }
    );
    const km = new KMeans({ nClusters: 2, randomState: 1 }).fit(Y);
    expect(km.clusterCenters.dtype).toBe("float64");
    const centers = (km.clusterCenters.toArray() as number[][]).sort(
      (a, b) => (a[0] as number) - (b[0] as number)
    );
    expect(centers[0]?.[0]).toBeCloseTo(0.2, 15);
    expect(centers[1]?.[1]).toBeCloseTo(10.45, 14);
    expect(km.labels.dtype).toBe("int32");
    expect(km.predict(Y).dtype).toBe("int32");
  });

  it("stops by a variance-relative tolerance, independent of the data scale", () => {
    let s = 3;
    const rand = (): number => {
      s = (s * 16807) % 2147483647;
      return s / 2147483647;
    };
    const base = Array.from({ length: 60 }, () => [
      rand() * 3 + (rand() > 0.5 ? 2 : 0),
      rand() * 3,
    ]);
    const runs = [1e-3, 1, 1e4].map((scale) => {
      const Y = tensor(
        base.map((r) => r.map((v) => v * scale)),
        { dtype: "float64" }
      );
      const km = new KMeans({
        nClusters: 4,
        randomState: 2,
        nInit: 1,
        algorithm: "lloyd",
        init: "random",
      }).fit(Y);
      return { nIter: km.nIter, labels: flat(km.labels) };
    });
    // With an absolute tolerance the smallest scale used to stop after 2 iterations instead of 9.
    expect(runs[0]?.nIter).toBe(runs[1]?.nIter);
    expect(runs[2]?.nIter).toBe(runs[1]?.nIter);
    expect(runs[0]?.labels).toEqual(runs[1]?.labels);
    expect(runs[2]?.labels).toEqual(runs[1]?.labels);
  });

  it("moves empty clusters to the farthest points instead of leaving them stale", () => {
    const D = tensor(
      [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [5, 5],
        [10, 10],
      ],
      { dtype: "float64" }
    );
    for (const algorithm of ["lloyd", "elkan"] as const) {
      for (let seed = 0; seed < 25; seed++) {
        const km = new KMeans({
          nClusters: 3,
          randomState: seed,
          nInit: 1,
          init: "random",
          algorithm,
        }).fit(D);
        // Three distinct locations, so the optimum has zero inertia.
        expect(km.inertia).toBeLessThan(1e-12);
      }
    }
  });

  it("k-means++ never repeats an existing location when distinct points remain", () => {
    const D = tensor(
      [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [2, 2],
      ],
      { dtype: "float64" }
    );
    for (let seed = 0; seed < 25; seed++) {
      const km = new KMeans({ nClusters: 3, randomState: seed, nInit: 1 }).fit(D);
      expect(km.inertia).toBeLessThan(1e-12);
    }
  });

  it("warns when duplicates leave fewer distinct clusters than requested", () => {
    const D = tensor(
      [
        [1, 1],
        [1, 1],
        [1, 1],
        [5, 5],
      ],
      { dtype: "float64" }
    );
    const warnings = catchWarnings(() => {
      const km = new KMeans({ nClusters: 3, randomState: 1 }).fit(D);
      expect(flat(km.predict(D))).toHaveLength(4);
    });
    expect(warnings.some((w) => w.category === "ConvergenceWarning")).toBe(true);
  });

  it("is reproducible for a seed and accepts negative or fractional seeds", () => {
    for (const randomState of [-5, 0.5, 42]) {
      const a = new KMeans({ nClusters: 3, randomState }).fit(X);
      const b = new KMeans({ nClusters: 3, randomState }).fit(X);
      expect(flat(a.clusterCenters)).toEqual(flat(b.clusterCenters));
      expect(flat(a.clusterCenters).every(Number.isFinite)).toBe(true);
      expect(a.inertia).toBeCloseTo(4.0132, 10);
    }
  });

  it("random init picks distinct samples even when nClusters equals n_samples", () => {
    const D = tensor([[0], [1], [2], [3]], { dtype: "float64" });
    const km = new KMeans({ nClusters: 4, init: "random", randomState: 3, nInit: 1 }).fit(D);
    expect(km.inertia).toBe(0);
    expect(new Set(flat(km.labels)).size).toBe(4);
  });

  it("warmStart restarts from the previous centroids and validates their shape", () => {
    const km = new KMeans({ nClusters: 3, randomState: 0, warmStart: true }).fit(X);
    const first = flat(km.clusterCenters);
    km.fit(X);
    expect(km.nIter).toBeLessThanOrEqual(2);
    expectClose(flat(km.clusterCenters), first, 1e-12);

    km.setParams({ nClusters: 2 });
    expect(() => km.fit(X)).toThrow(ShapeError);
    km.setParams({ nClusters: 3 });
    expect(() =>
      km.fit(
        tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
          { dtype: "float64" }
        )
      )
    ).toThrow(ShapeError);
  });

  it("setParams accepts every key that getParams returns (including warmStart)", () => {
    const km = new KMeans({ nClusters: 3, warmStart: true, randomState: 7 });
    const other = new KMeans();
    expect(() => other.setParams(km.getParams())).not.toThrow();
    expect(other.getParams()).toEqual(km.getParams());
    expect(() => other.setParams({ warmStart: "yes" })).toThrow(InvalidParameterError);
    expect(() => other.setParams({ tol: Number.POSITIVE_INFINITY })).toThrow(InvalidParameterError);
    expect(() => new KMeans({ warmStart: 1 as unknown as boolean })).toThrow(InvalidParameterError);
  });

  it("a failed fit leaves the fitted model usable", () => {
    const km = new KMeans({ nClusters: 3, randomState: 0 }).fit(
      tensor(
        KM_X.map((r) => [r[0] as number, r[1] as number, 0]),
        { dtype: "float64" }
      )
    );
    expect(() => km.fit(tensor([[1, 2]], { dtype: "float64" }))).toThrow(InvalidParameterError);
    // The 3-feature model must still validate against 3 features, not the failed 2-feature input.
    expect(() => km.predict(tensor([[1, 2]], { dtype: "float64" }))).toThrow(ShapeError);
    expect(km.predict(tensor([[0, 0, 0]], { dtype: "float64" })).size).toBe(1);
    expect(km.nFeaturesIn).toBe(3);
  });

  it("fitTransform equals fit then transform, and clone is unfitted", () => {
    const km = new KMeans({ nClusters: 3, randomState: 0 });
    const t = km.fitTransform(X);
    expect(t.shape).toEqual([15, 3]);
    expectClose(flat(t), flat(km.transform(X)), 0);
    const copy = km.clone();
    expect(copy.getParams()).toEqual(km.getParams());
    expect(() => copy.labels).toThrow(NotFittedError);
  });

  it("accepts the scikit-learn spelling k-means++ and normalizes it", () => {
    const km = new KMeans({ nClusters: 3, init: "k-means++", randomState: 0 });
    expect(km.getParams().init).toBe("kmeans++");
    km.setParams({ init: "random" });
    km.setParams({ init: "k-means++" });
    expect(km.getParams().init).toBe("kmeans++");
    expect(() => km.setParams({ init: "forgy" })).toThrow(InvalidParameterError);
  });

  it("predict and transform use the fitted number of clusters after setParams", () => {
    const km = new KMeans({ nClusters: 3, randomState: 0 }).fit(X);
    const before = flat(km.predict(X));
    km.setParams({ nClusters: 6 });
    expect(flat(km.predict(X))).toEqual(before);
    expect(km.transform(X).shape).toEqual([15, 3]);
    expect(km.score(X)).toBeCloseTo(-4.0132, 10);
  });

  it("does not modify the input tensor", () => {
    const Y = tensor(KM_X, { dtype: "float64" });
    const before = flat(Y);
    new KMeans({ nClusters: 3, randomState: 0 }).fit(Y);
    expect(flat(Y)).toEqual(before);
  });
});

// ---------------------------------------------------------------------------------------------
// DBSCAN
// ---------------------------------------------------------------------------------------------

const DB_X = [
  [-0.22, 0.32],
  [-0.5, 0.16],
  [-0.62, -0.2],
  [-0.36, 0.44],
  [0.53, -0.1],
  [0.25, -0.05],
  [0.17, -0.23],
  [-0.51, -0.54],
  [0.11, 0.67],
  [0.08, -0.16],
  [3.57, 3.07],
  [3.03, 3.08],
  [2.96, 2.91],
  [2.57, 3.15],
  [2.97, 3.36],
  [2.89, 2.43],
  [2.97, 3.51],
  [2.88, 2.73],
  [2.64, 2.68],
  [2.91, 2.65],
  [1.14, 3.42],
  [3.57, 1.66],
  [1.22, 3.45],
  [4.21, 2.72],
];

describe("DBSCAN", () => {
  const X = tensor(DB_X, { dtype: "float64" });

  it("matches scikit-learn labels and returns sorted core sample indices", () => {
    const db = new DBSCAN({ eps: 0.6, minSamples: 3 }).fit(X);
    expect(flat(db.labels)).toEqual([
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1,
    ]);
    // The previous implementation listed core samples in discovery order, not sorted.
    expect(db.coreIndices).toEqual([0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
    expect(db.nClusters).toBe(2);
  });

  it("predict validates the feature count, NaN and dimensionality", () => {
    const db = new DBSCAN({ eps: 0.6, minSamples: 3 }).fit(X);
    expect(() => db.predict(tensor([[1, 2, 3]], { dtype: "float64" }))).toThrow(ShapeError);
    expect(() => db.predict(tensor([[Number.NaN, 0]], { dtype: "float64" }))).toThrow(
      DataValidationError
    );
    expect(() => db.predict(tensor([1, 2], { dtype: "float64" }))).toThrow(ShapeError);
  });

  it("predict reproduces the training label of every core sample", () => {
    const db = new DBSCAN({ eps: 0.6, minSamples: 3 }).fit(X);
    const train = flat(db.labels);
    const pred = flat(db.predict(X));
    for (const idx of db.coreIndices) expect(pred[idx]).toBe(train[idx]);
    // A far point is noise.
    expect(flat(db.predict(tensor([[100, 100]], { dtype: "float64" })))).toEqual([-1]);
  });

  it("predict uses the same <= eps rule as fit at the boundary", () => {
    // Distance between the points is exactly eps = 0.3 (0.3 * 0.3 === 0.09 in floating point).
    const P = tensor(
      [
        [0, 0],
        [0.3, 0],
        [0, 0.3],
      ],
      { dtype: "float64" }
    );
    const db = new DBSCAN({ eps: 0.3, minSamples: 2 }).fit(P);
    const trainLabels = flat(db.labels);
    expect(trainLabels.every((l) => l === 0)).toBe(true);
    expect(flat(db.predict(P))).toEqual(trainLabels);
  });

  it("manhattan metric is honoured by fit and predict", () => {
    const P = tensor(
      [
        [0, 0],
        [0.4, 0.4],
        [0.8, 0],
        [5, 5],
      ],
      { dtype: "float64" }
    );
    const euclid = new DBSCAN({ eps: 0.6, minSamples: 2 }).fit(P);
    const manhattan = new DBSCAN({ eps: 0.6, minSamples: 2, metric: "manhattan" }).fit(P);
    expect(flat(euclid.labels)).toEqual([0, 0, 0, -1]);
    expect(flat(manhattan.labels)).toEqual([-1, -1, -1, -1]);
    expect(flat(manhattan.predict(tensor([[0.1, 0.1]], { dtype: "float64" })))).toEqual([-1]);
  });

  it("is not affected by later edits of the training tensor", () => {
    const Y = tensor(DB_X, { dtype: "float64" });
    const db = new DBSCAN({ eps: 0.6, minSamples: 3 }).fit(Y);
    const before = flat(db.predict(Y));
    (Y.data as Float64Array).fill(1000);
    expect(flat(db.predict(X))).toEqual(before);
  });

  it("validates eps consistently in the constructor and setParams", () => {
    expect(() => new DBSCAN({ eps: Number.POSITIVE_INFINITY })).toThrow(InvalidParameterError);
    expect(() => new DBSCAN({ eps: Number.NaN })).toThrow(InvalidParameterError);
    const db = new DBSCAN();
    expect(() => db.setParams({ eps: Number.POSITIVE_INFINITY })).toThrow(InvalidParameterError);
    expect(() => db.setParams({ eps: Number.NaN })).toThrow(InvalidParameterError);
  });

  it("predict keeps the eps and metric of the last fit after setParams", () => {
    const db = new DBSCAN({ eps: 0.6, minSamples: 3 }).fit(X);
    const probe = tensor([[1.5, 1.5]], { dtype: "float64" });
    expect(flat(db.predict(probe))).toEqual([-1]);
    db.setParams({ eps: 100, metric: "manhattan" });
    expect(flat(db.predict(probe))).toEqual([-1]);
  });

  it("handles a border point seen first as noise and claims it for the cluster", () => {
    // Index 0 is a border point (2 neighbors incl. itself) visited before the core point 1.
    const P = tensor([[0], [0.5], [1.0], [3]], { dtype: "float64" });
    const db = new DBSCAN({ eps: 0.55, minSamples: 3 }).fit(P);
    expect(flat(db.labels)).toEqual([0, 0, 0, -1]);
    expect(db.coreIndices).toEqual([1]);
  });
});

// ---------------------------------------------------------------------------------------------
// AgglomerativeClustering
// ---------------------------------------------------------------------------------------------

const AG_X = [
  [-0.1, -0.89],
  [0.33, 0.45],
  [0.21, -0.46],
  [-0.1, -0.3],
  [4.85, 5.65],
  [5.76, 5.33],
  [5.27, 5.34],
  [-0.01, 5.96],
  [-0.34, 5.97],
  [1.13, 6.43],
];

describe("AgglomerativeClustering", () => {
  const X = tensor(AG_X, { dtype: "float64" });
  const expectedLabels = [1, 1, 1, 1, 2, 2, 2, 0, 0, 0];
  const reference = {
    ward: {
      dist: [
        0.3301514804, 0.3488552709, 0.4901020302, 0.6154943812, 0.8496666013, 1.2884357441,
        1.5996874695, 8.797188945, 14.5676028685,
      ],
      children: [
        [7, 8],
        [2, 3],
        [5, 6],
        [0, 11],
        [4, 12],
        [1, 13],
        [9, 10],
        [14, 16],
        [15, 17],
      ],
    },
    complete: {
      dist: [
        0.3301514804, 0.3488552709, 0.4901020302, 0.59, 0.9646242792, 1.407302384, 1.5402921801,
        6.1334818823, 8.5456421643,
      ],
      children: [
        [7, 8],
        [2, 3],
        [5, 6],
        [0, 11],
        [4, 12],
        [1, 13],
        [9, 10],
        [14, 16],
        [15, 17],
      ],
    },
    average: {
      dist: [
        0.3301514804, 0.3488552709, 0.4901020302, 0.5600471656, 0.7433198023, 1.0632344538,
        1.3866888704, 5.0899892461, 7.1089224135,
      ],
      children: [
        [7, 8],
        [2, 3],
        [5, 6],
        [0, 11],
        [4, 12],
        [1, 13],
        [9, 10],
        [14, 16],
        [15, 17],
      ],
    },
    single: {
      dist: [
        0.3301514804, 0.3488552709, 0.4901020302, 0.5220153254, 0.5300943312, 0.8645229899,
        1.2330855607, 3.8008946315, 5.5204800516,
      ],
      // SciPy lists the children of single linkage in MST order; compare as unordered pairs.
      children: [
        [7, 8],
        [2, 3],
        [5, 6],
        [4, 12],
        [0, 11],
        [1, 14],
        [9, 10],
        [13, 16],
        [15, 17],
      ],
    },
  } as const;

  for (const linkage of ["ward", "complete", "average", "single"] as const) {
    it(`matches scikit-learn/SciPy labels, merge distances and dendrogram (${linkage})`, () => {
      const agg = new AgglomerativeClustering({ nClusters: 3, linkage }).fit(X);
      expect(flat(agg.labels)).toEqual(expectedLabels);
      expectClose(flat(agg.distances), reference[linkage].dist, 1e-9);
      const children = (agg.children.toArray() as number[][]).map((p) =>
        [...p].sort((a, b) => a - b)
      );
      expect(children).toEqual(
        reference[linkage].children.map((p) => [...p].sort((a, b) => a - b))
      );
      expect(agg.children.shape).toEqual([9, 2]);
      expect(agg.nLeaves).toBe(10);
      expect(agg.nClusters).toBe(3);
    });
  }

  it("supports distanceThreshold like scikit-learn", () => {
    const agg = new AgglomerativeClustering({ distanceThreshold: 2.5, linkage: "average" }).fit(X);
    expect(flat(agg.labels)).toEqual(expectedLabels);
    expect(agg.nClusters).toBe(3);
    expect(agg.getParams().nClusters).toBeNull();
    // A zero threshold keeps every sample apart; a huge one merges everything.
    expect(new AgglomerativeClustering({ distanceThreshold: 0 }).fit(X).nClusters).toBe(10);
    expect(new AgglomerativeClustering({ distanceThreshold: 1e9 }).fit(X).nClusters).toBe(1);
  });

  it("requires exactly one of nClusters and distanceThreshold", () => {
    expect(() => new AgglomerativeClustering({ nClusters: 2, distanceThreshold: 1 })).toThrow(
      InvalidParameterError
    );
    expect(() => new AgglomerativeClustering({ nClusters: null })).toThrow(InvalidParameterError);
    expect(() => new AgglomerativeClustering({ distanceThreshold: -1 })).toThrow(
      InvalidParameterError
    );
  });

  it("validates linkage in the constructor", () => {
    expect(() => new AgglomerativeClustering({ linkage: "centroid" as "ward" })).toThrow(
      InvalidParameterError
    );
  });

  it("cluster centers are float64 centroids ordered by label", () => {
    const agg = new AgglomerativeClustering({ nClusters: 3 }).fit(X);
    const centers = agg.clusterCenters;
    expect(centers.dtype).toBe("float64");
    expect(centers.shape).toEqual([3, 2]);
    // Label 0 is the cluster of the last three samples.
    expectClose(
      flat(centers).slice(0, 2),
      [(-0.01 - 0.34 + 1.13) / 3, (5.96 + 5.97 + 6.43) / 3],
      1e-12
    );
  });

  it("predict validates features and is independent of later edits of X", () => {
    const Y = tensor(AG_X, { dtype: "float64" });
    const agg = new AgglomerativeClustering({ nClusters: 3 }).fit(Y);
    expect(() => agg.predict(tensor([[1, 2, 3]], { dtype: "float64" }))).toThrow(ShapeError);
    expect(() => agg.predict(tensor([[Number.NaN, 1]], { dtype: "float64" }))).toThrow(
      DataValidationError
    );
    const probe = tensor(
      [
        [0, 0],
        [5, 5],
        [0, 6],
      ],
      { dtype: "float64" }
    );
    const before = flat(agg.predict(probe));
    expect(before).toEqual([1, 2, 0]);
    (Y.data as Float64Array).fill(-50);
    expect(flat(agg.predict(probe))).toEqual(before);
  });

  it("handles a single sample and two samples", () => {
    const one = new AgglomerativeClustering({ nClusters: 1 }).fit(tensor([[1, 2]]));
    expect(flat(one.labels)).toEqual([0]);
    expect(one.children.shape).toEqual([0, 2]);
    const two = new AgglomerativeClustering({ nClusters: 2 }).fit(
      tensor(
        [
          [0, 0],
          [1, 1],
        ],
        { dtype: "float64" }
      )
    );
    expect(canon(flat(two.labels))).toEqual([0, 1]);
  });

  it("builds the tree for a few thousand samples quickly (no cubic cost per merge)", () => {
    let s = 9;
    const rand = (): number => {
      s = (s * 16807) % 2147483647;
      return s / 2147483647;
    };
    const n = 1200;
    const rows = Array.from({ length: n }, (_, i) => [
      (i % 4) * 10 + rand(),
      Math.floor(i / 4) * 0 + rand(),
    ]);
    const start = Date.now();
    const agg = new AgglomerativeClustering({ nClusters: 4 }).fit(
      tensor(rows, { dtype: "float64" })
    );
    expect(Date.now() - start).toBeLessThan(5000);
    expect(new Set(flat(agg.labels)).size).toBe(4);
  });

  it("setParams round-trips getParams", () => {
    const agg = new AgglomerativeClustering({ nClusters: 3, linkage: "single" });
    const other = new AgglomerativeClustering();
    other.setParams(agg.getParams());
    expect(other.getParams()).toEqual(agg.getParams());
    other.setParams({ nClusters: null, distanceThreshold: 2 });
    expect(other.getParams().distanceThreshold).toBe(2);
  });
});

// ---------------------------------------------------------------------------------------------
// AffinityPropagation
// ---------------------------------------------------------------------------------------------

const AP_X = [
  [-0.015],
  [4.295],
  [5.212],
  [4.472],
  [-1.165],
  [2.161],
  [-2.654],
  [2.889],
  [0.493],
  [0.126],
  [7.779],
  [6.136],
  [1.698],
  [7.732],
  [9.818],
];

describe("AffinityPropagation", () => {
  const X = tensor(AP_X, { dtype: "float64" });

  it("matches scikit-learn: default preference, exemplars, labels and iteration count", () => {
    // The default preference is the median of the whole similarity matrix (diagonal
    // included); using the off-diagonal median found 3 clusters here instead of 4.
    const ap = new AffinityPropagation().fit(X);
    expect(flat(ap.labels)).toEqual([1, 0, 0, 0, 1, 3, 1, 3, 3, 1, 2, 0, 3, 2, 2]);
    expect(Array.from(ap.clusterCentersIndices)).toEqual([2, 4, 10, 12]);
    expect(ap.nIter).toBe(22);
    expect(ap.converged).toBe(true);
    expect(flat(ap.clusterCenters)).toEqual([5.212, -1.165, 7.779, 1.698]);
    expect(ap.clusterCenters.shape).toEqual([4, 1]);
  });

  it("predict assigns the nearest exemplar", () => {
    const ap = new AffinityPropagation().fit(X);
    expect(flat(ap.predict(tensor([[5], [-2], [8], [1.5]], { dtype: "float64" })))).toEqual([
      0, 1, 2, 3,
    ]);
    expect(() => ap.predict(tensor([[1, 2]], { dtype: "float64" }))).toThrow(ShapeError);
  });

  it("supports a precomputed similarity matrix", () => {
    const n = AP_X.length;
    const S: number[][] = [];
    for (let i = 0; i < n; i++) {
      S.push(AP_X.map((row) => -(((AP_X[i]?.[0] as number) - (row[0] as number)) ** 2)));
    }
    const ap = new AffinityPropagation({ affinity: "precomputed" }).fit(
      tensor(S, { dtype: "float64" })
    );
    expect(flat(ap.labels)).toEqual([1, 0, 0, 0, 1, 3, 1, 3, 3, 1, 2, 0, 3, 2, 2]);
    expect(Array.from(ap.clusterCentersIndices)).toEqual([2, 4, 10, 12]);
    expect(() => ap.predict(X)).toThrow(InvalidParameterError);
    expect(() => ap.clusterCenters).toThrow(DataValidationError);
    expect(() =>
      new AffinityPropagation({ affinity: "precomputed" }).fit(
        tensor([[1, 2, 3]], { dtype: "float64" })
      )
    ).toThrow(ShapeError);
  });

  it("returns labels of -1 and no centers when no exemplar emerges (like scikit-learn)", () => {
    const warnings = catchWarnings(() => {
      const ap = new AffinityPropagation({ maxIter: 1 }).fit(X);
      expect(flat(ap.labels)).toEqual(new Array(15).fill(-1));
      expect(ap.clusterCentersIndices.length).toBe(0);
      expect(ap.converged).toBe(false);
      expect(ap.nIter).toBe(1);
      expect(ap.clusterCenters.shape).toEqual([0, 1]);
      expect(flat(ap.predict(tensor([[1], [2]], { dtype: "float64" })))).toEqual([-1, -1]);
    });
    expect(warnings.filter((w) => w.category === "ConvergenceWarning").length).toBeGreaterThan(0);
  });

  it("handles identical samples and a single sample like scikit-learn", () => {
    const same = tensor(
      [
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
      ],
      { dtype: "float64" }
    );
    catchWarnings(() => {
      const one = new AffinityPropagation().fit(same);
      expect(flat(one.labels)).toEqual([0, 0, 0, 0]);
      expect(Array.from(one.clusterCentersIndices)).toEqual([0]);

      const many = new AffinityPropagation({ preference: 1 }).fit(same);
      expect(flat(many.labels)).toEqual([0, 1, 2, 3]);
      expect(Array.from(many.clusterCentersIndices)).toEqual([0, 1, 2, 3]);

      const single = new AffinityPropagation().fit(tensor([[1, 2]], { dtype: "float64" }));
      expect(flat(single.labels)).toEqual([0]);
      expect(Array.from(single.clusterCentersIndices)).toEqual([0]);
    });
  });

  it("scales to a few hundred samples (message passing is O(n^2) per iteration)", () => {
    let s = 5;
    const rand = (): number => {
      s = (s * 16807) % 2147483647;
      return s / 2147483647;
    };
    const rows = Array.from({ length: 400 }, (_, i) => [(i % 3) * 20 + rand(), rand()]);
    const start = Date.now();
    const ap = new AffinityPropagation({ damping: 0.8 }).fit(tensor(rows, { dtype: "float64" }));
    expect(Date.now() - start).toBeLessThan(5000);
    expect(ap.clusterCentersIndices.length).toBeGreaterThanOrEqual(3);
  });

  it("validates every option, including NaN and convergenceIter", () => {
    expect(() => new AffinityPropagation({ damping: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new AffinityPropagation({ convergenceIter: 0 })).toThrow(InvalidParameterError);
    expect(() => new AffinityPropagation({ convergenceIter: 1.5 })).toThrow(InvalidParameterError);
    expect(() => new AffinityPropagation({ preference: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => new AffinityPropagation({ affinity: "cosine" as "euclidean" })).toThrow(
      InvalidParameterError
    );
    const ap = new AffinityPropagation();
    expect(() => ap.setParams({ damping: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => ap.setParams({ preference: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => ap.setParams({ randomState: Number.NaN })).toThrow(InvalidParameterError);
    const other = new AffinityPropagation();
    other.setParams({ ...ap.getParams(), preference: undefined });
    expect(other.getParams()).toEqual(ap.getParams());
  });

  it("predict depends on how the model was fitted, not on the current affinity option", () => {
    const S = AP_X.map((a) => AP_X.map((b) => -(((a[0] as number) - (b[0] as number)) ** 2)));
    const pre = new AffinityPropagation({ affinity: "precomputed" }).fit(
      tensor(S, { dtype: "float64" })
    );
    pre.setParams({ affinity: "euclidean" });
    expect(() => pre.predict(X)).toThrow(InvalidParameterError);

    const euclid = new AffinityPropagation().fit(X);
    euclid.setParams({ affinity: "precomputed" });
    expect(flat(euclid.predict(tensor([[5], [-2]], { dtype: "float64" })))).toEqual([0, 1]);
  });

  it("refit replaces the previous model (centers for a different feature count)", () => {
    const ap = new AffinityPropagation();
    ap.fit(X);
    ap.fit(
      tensor(
        [
          [0, 0],
          [0.1, 0],
          [10, 10],
          [10.1, 10],
        ],
        { dtype: "float64" }
      )
    );
    expect(ap.clusterCenters.shape[1]).toBe(2);
    expect(() => ap.predict(X)).toThrow(ShapeError);
  });
});

// ---------------------------------------------------------------------------------------------
// GaussianMixture
// ---------------------------------------------------------------------------------------------

const GM_X = [
  [0.87, -0.29],
  [-0.24, -2.65],
  [-0.0, -0.32],
  [-0.27, 0.32],
  [0.21, -1.07],
  [-0.44, -0.48],
  [0.34, 0.56],
  [-0.65, -1.12],
  [0.37, 1.57],
  [-0.02, -0.68],
  [0.55, -0.31],
  [0.36, 1.55],
  [6.63, 3.03],
  [6.73, 2.74],
  [5.82, 2.77],
  [5.8, 2.81],
  [5.81, 2.85],
  [6.09, 3.03],
  [6.3, 3.56],
  [4.45, 3.52],
  [5.76, 2.51],
  [5.83, 3.04],
  [7.07, 2.58],
  [6.22, 3.05],
];

describe("GaussianMixture", () => {
  const X = tensor(GM_X, { dtype: "float64" });

  const reference = {
    full: {
      cov: [
        0.177951, 0.2039166667, 0.2039166667, 1.2475065556, 0.3997530833, -0.0836854167,
        -0.0836854167, 0.0969864167,
      ],
      score: -2.2405628510058615,
      bic: 142.50560898210875,
      aic: 129.54701684828134,
      tol: 1e-8,
    },
    tied: {
      cov: [0.2888520417, 0.060115625, 0.060115625, 0.6722464861],
      score: -2.7021430708811742,
      bic: 155.12729804507993,
      aic: 145.70286740229636,
      tol: 1e-8,
    },
    diag: {
      cov: [0.177951, 1.2475065556, 0.3997530833, 0.0969864167],
      score: -2.342221237948314,
      bic: 141.02910389465058,
      aic: 130.42661942151906,
      tol: 1e-8,
    },
    spherical: {
      cov: [0.712728789, 0.2483697492],
      score: -2.6652760107149014,
      bic: 150.17962532675088,
      aic: 141.93324851431527,
      tol: 1e-6,
    },
  } as const;

  for (const covarianceType of ["full", "tied", "diag", "spherical"] as const) {
    it(`matches scikit-learn weights, means, covariances, score, bic and aic (${covarianceType})`, () => {
      const gm = new GaussianMixture({
        nComponents: 2,
        covarianceType,
        tol: 1e-10,
        maxIter: 500,
        randomState: 0,
      }).fit(X);
      const ref = reference[covarianceType];
      expect(gm.converged).toBe(true);

      const means = gm.means.toArray() as number[][];
      const order = [0, 1].sort((a, b) => (means[a]?.[0] ?? 0) - (means[b]?.[0] ?? 0));
      expectClose(
        order.flatMap((i) => means[i] as number[]),
        [0.09, -0.2433333333, 6.0425, 2.9575],
        1e-6
      );
      const weights = flat(gm.weights);
      expectClose(
        order.map((i) => weights[i] as number),
        [0.5, 0.5],
        1e-6
      );

      const covs = gm.covariances;
      const covFlat = flat(covs);
      const perComp = covFlat.length / (covarianceType === "tied" ? 1 : 2);
      const orderedCov =
        covarianceType === "tied"
          ? covFlat
          : order.flatMap((i) => covFlat.slice(i * perComp, (i + 1) * perComp));
      expectClose(orderedCov, ref.cov, ref.tol);

      expect(Math.abs(gm.score(X) - ref.score)).toBeLessThan(ref.tol);
      expect(Math.abs(gm.bic(X) - ref.bic)).toBeLessThan(ref.tol * 100);
      expect(Math.abs(gm.aic(X) - ref.aic)).toBeLessThan(ref.tol * 100);
    });
  }

  it("exposes covariance shapes like scikit-learn", () => {
    const shapes = { full: [2, 2, 2], tied: [2, 2], diag: [2, 2], spherical: [2] } as const;
    for (const covarianceType of ["full", "tied", "diag", "spherical"] as const) {
      const gm = new GaussianMixture({ nComponents: 2, covarianceType, randomState: 0 }).fit(X);
      expect(gm.covariances.shape).toEqual([...shapes[covarianceType]]);
      expect(gm.covariances.dtype).toBe("float64");
    }
  });

  it("adds regCovar to the variances (scikit-learn) instead of using it as a floor", () => {
    const gm = new GaussianMixture({
      nComponents: 2,
      tol: 1e-10,
      maxIter: 500,
      randomState: 0,
      regCovar: 0.1,
    }).fit(X);
    const means = gm.means.toArray() as number[][];
    const order = [0, 1].sort((a, b) => (means[a]?.[0] ?? 0) - (means[b]?.[0] ?? 0));
    const cov = flat(gm.covariances);
    expectClose(
      order.flatMap((i) => cov.slice(i * 2, i * 2 + 2)),
      [0.27795, 1.3475055556, 0.4997520833, 0.1969854167],
      1e-8
    );
  });

  it("reports the mean log-likelihood per sample and compares it against tol per sample", () => {
    const gm = new GaussianMixture({
      nComponents: 2,
      tol: 1e-10,
      maxIter: 500,
      randomState: 0,
    }).fit(X);
    expect(gm.lowerBound).toBeCloseTo(-2.342221237948314, 6);
    expect(gm.nIter).toBe(2);
    expect(gm.nIter).toBeGreaterThanOrEqual(1);
  });

  it("predictProba returns float64 rows that sum to 1 without underflow", () => {
    const gm = new GaussianMixture({
      nComponents: 2,
      tol: 1e-10,
      maxIter: 500,
      randomState: 0,
    }).fit(X);
    const probe = tensor(
      [
        [0.5, 0.5],
        [3.0, 1.5],
        [20.0, -20.0],
      ],
      { dtype: "float64" }
    );
    const proba = gm.predictProba(probe);
    expect(proba.dtype).toBe("float64");
    const means = gm.means.toArray() as number[][];
    const first = (means[0]?.[0] as number) < (means[1]?.[0] as number) ? 0 : 1;
    const p = flat(proba);
    const ordered = [0, 1, 2].flatMap((r) => [
      p[r * 2 + first] as number,
      p[r * 2 + 1 - first] as number,
    ]);
    expectClose(ordered, [1, 0, 0.033756681205, 0.966243318795, 1, 0], 1e-8);
    for (let r = 0; r < 3; r++) {
      expect((p[r * 2] as number) + (p[r * 2 + 1] as number)).toBeCloseTo(1, 14);
    }
    const ss = flat(gm.scoreSamples(probe));
    expectClose(ss, [-2.4722546926, -23.401359097, -1272.0333997989], 1e-6);
  });

  it("cluster centers keep float64 precision", () => {
    const gm = new GaussianMixture({ nComponents: 2, randomState: 0 }).fit(X);
    expect(gm.clusterCenters.dtype).toBe("float64");
    expect(gm.clusterCenters.shape).toEqual([2, 2]);
    expect(gm.labels.dtype).toBe("int32");
  });

  it("fit labels agree with predict on the training data", () => {
    for (const covarianceType of ["full", "diag"] as const) {
      const gm = new GaussianMixture({ nComponents: 3, covarianceType, randomState: 4 }).fit(X);
      expect(flat(gm.predict(X))).toEqual(flat(gm.labels));
    }
  });

  it("is reproducible for a seed for every initialization method", () => {
    for (const initParams of ["kmeans", "random", "randomFromData"] as const) {
      const a = new GaussianMixture({ nComponents: 2, initParams, randomState: -3, nInit: 2 }).fit(
        X
      );
      const b = new GaussianMixture({ nComponents: 2, initParams, randomState: -3, nInit: 2 }).fit(
        X
      );
      expect(a.lowerBound).toBe(b.lowerBound);
      expect(flat(a.clusterCenters)).toEqual(flat(b.clusterCenters));
    }
  });

  it("fails with a ConvergenceError when a covariance is singular and regCovar is 0", () => {
    // Perfectly collinear samples give a rank-1 covariance matrix.
    const line = tensor(
      [1, 2, 3, 4, 5, 6].map((i) => [i, 2 * i]),
      { dtype: "float64" }
    );
    const singular = new GaussianMixture({ nComponents: 1, covarianceType: "full", regCovar: 0 });
    expect(() => singular.fit(line)).toThrow(ConvergenceError);
    expect(() => singular.fit(line)).toThrow(/ill-defined/);
    // With the default regularization the same data fits.
    const regularized = new GaussianMixture({ nComponents: 1, covarianceType: "full" }).fit(line);
    expect(regularized.converged).toBe(true);
  });

  it("warns when EM does not converge", () => {
    const warnings = catchWarnings(() => {
      new GaussianMixture({ nComponents: 2, maxIter: 1, tol: 1e-12, randomState: 0 }).fit(X);
    });
    expect(warnings.some((w) => w.category === "ConvergenceWarning")).toBe(true);
  });

  it("validates every constructor option (previously only nComponents was checked)", () => {
    expect(() => new GaussianMixture({ maxIter: 0 })).toThrow(InvalidParameterError);
    expect(() => new GaussianMixture({ tol: -1 })).toThrow(InvalidParameterError);
    expect(() => new GaussianMixture({ tol: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new GaussianMixture({ nInit: 0 })).toThrow(InvalidParameterError);
    expect(() => new GaussianMixture({ regCovar: -1 })).toThrow(InvalidParameterError);
    expect(() => new GaussianMixture({ randomState: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new GaussianMixture({ covarianceType: "x" as "full" })).toThrow(
      InvalidParameterError
    );
    expect(() => new GaussianMixture({ initParams: "x" as "kmeans" })).toThrow(
      InvalidParameterError
    );
    const gm = new GaussianMixture();
    expect(() => gm.setParams({ covarianceType: "x" })).toThrow(InvalidParameterError);
    expect(() => gm.setParams({ tol: Number.NaN })).toThrow(InvalidParameterError);
    const other = new GaussianMixture();
    other.setParams({ ...gm.getParams(), nComponents: 2 });
    expect(other.getParams().nComponents).toBe(2);
  });

  it("is still tagged as a clusterer although it has scoreSamples", () => {
    const tags = getEstimatorTags(new GaussianMixture());
    expect(tags.estimatorType).toBe("clusterer");
    expect(tags.requiresY).toBe(false);
    expect(tags.hasPredictProba).toBe(true);
  });

  it("changing hyperparameters after fit does not break the fitted model", () => {
    const gm = new GaussianMixture({ nComponents: 2, randomState: 0 }).fit(X);
    const before = flat(gm.predictProba(X));
    gm.setParams({ nComponents: 5, covarianceType: "full" });
    expect(flat(gm.predictProba(X))).toEqual(before);
    expect(gm.weights.shape).toEqual([2]);
  });

  it("throws NotFittedError before fit for every accessor", () => {
    const gm = new GaussianMixture({ nComponents: 2 });
    for (const read of [
      () => gm.weights,
      () => gm.means,
      () => gm.covariances,
      () => gm.converged,
      () => gm.nIter,
      () => gm.lowerBound,
      () => gm.score(X),
      () => gm.bic(X),
      () => gm.scoreSamples(X),
    ]) {
      expect(read).toThrow(NotFittedError);
    }
  });
});

// ---------------------------------------------------------------------------------------------
// Birch
// ---------------------------------------------------------------------------------------------

const BI_X = [
  [0.13, -0.1],
  [0.73, -0.08],
  [0.03, 0.47],
  [-0.27, -0.18],
  [0.06, -0.1],
  [-0.36, -0.06],
  [-0.11, 0.18],
  [-0.5, -0.21],
  [0.35, 0.56],
  [-0.45, 0.19],
  [3.71, -0.26],
  [3.74, -0.13],
  [4.3, 0.21],
  [4.02, -0.11],
  [4.0, -0.03],
  [4.24, -0.19],
  [4.0, -0.03],
  [3.98, 0.07],
  [4.06, 0.4],
  [3.97, 0.47],
  [-0.09, 3.86],
  [0.03, 4.11],
  [0.08, 4.39],
  [0.34, 4.15],
  [-0.1, 3.97],
  [0.42, 4.07],
  [-0.39, 3.79],
  [-0.17, 4.35],
  [-0.03, 4.68],
  [0.2, 4.04],
];

describe("Birch", () => {
  const X = tensor(BI_X, { dtype: "float64" });

  it("matches scikit-learn: CF tree leaf entries and final clusters (multi-level tree)", () => {
    const wide = new Birch({ threshold: 0.4, branchingFactor: 3, nClusters: 3 }).fit(X);
    expect(wide.subclusterCenters.shape).toEqual([5, 2]);
    expect(canon(flat(wide.labels))).toEqual([
      ...new Array(10).fill(0),
      ...new Array(10).fill(1),
      ...new Array(10).fill(2),
    ]);

    // Branching factor 2 forces several node splits.
    const deep = new Birch({ threshold: 0.25, branchingFactor: 2, nClusters: 4 }).fit(X);
    expect(deep.subclusterCenters.shape).toEqual([11, 2]);
    expect(canon(flat(deep.labels))).toEqual([
      0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    ]);
  });

  it("limits the radius of an entry, not the distance to its centroid", () => {
    // Two points 0.8 apart form one entry of radius 0.4 <= 0.5. The old distance rule split them.
    const pair = tensor([[0], [0.8]], { dtype: "float64" });
    const birch = new Birch({ threshold: 0.5, nClusters: null }).fit(pair);
    expect(birch.subclusterCenters.shape).toEqual([1, 1]);
    expect(flat(birch.subclusterCenters)[0]).toBeCloseTo(0.4, 12);
    const apart = tensor([[0], [1.1]], { dtype: "float64" });
    expect(
      new Birch({ threshold: 0.5, nClusters: null }).fit(apart).subclusterCenters.shape
    ).toEqual([2, 1]);
  });

  it("is not fooled by a large constant offset in the data", () => {
    const offset = 1e9;
    const shifted = tensor(
      BI_X.map((r) => r.map((v) => v + offset)),
      { dtype: "float64" }
    );
    const plain = new Birch({ threshold: 0.4, branchingFactor: 3, nClusters: null }).fit(X);
    const moved = new Birch({ threshold: 0.4, branchingFactor: 3, nClusters: null }).fit(shifted);
    expect(moved.subclusterCenters.shape).toEqual(plain.subclusterCenters.shape);
    expect((moved.subclusterCenters.toArray() as number[][])[0]?.[0]).toBeGreaterThan(offset - 1);
  });

  it("nClusters null keeps every leaf entry as its own cluster", () => {
    const birch = new Birch({ threshold: 0.4, branchingFactor: 3, nClusters: null }).fit(X);
    const k = birch.subclusterCenters.shape[0] as number;
    expect(new Set(flat(birch.subclusterLabels)).size).toBe(k);
    expect(birch.clusterCenters.shape).toEqual([k, 2]);
    expect(birch.getParams().nClusters).toBeNull();
  });

  it("warns and returns fewer clusters when there are fewer entries than nClusters", () => {
    const warnings = catchWarnings(() => {
      const birch = new Birch({ nClusters: 6, threshold: 5 }).fit(X);
      expect(birch.clusterCenters.shape[0]).toBeLessThan(6);
      expect(birch.predict(X).size).toBe(30);
    });
    expect(warnings.some((w) => w.category === "ConvergenceWarning")).toBe(true);
  });

  it("cluster centers are count-weighted means of the samples in float64", () => {
    const birch = new Birch({ threshold: 0.4, branchingFactor: 3, nClusters: 3 }).fit(X);
    expect(birch.clusterCenters.dtype).toBe("float64");
    const labels = flat(birch.labels);
    const centers = birch.clusterCenters.toArray() as number[][];
    for (let c = 0; c < 3; c++) {
      const members = BI_X.filter((_, i) => labels[i] === c);
      const mx = members.reduce((a, r) => a + (r[0] as number), 0) / members.length;
      const my = members.reduce((a, r) => a + (r[1] as number), 0) / members.length;
      expect(centers[c]?.[0]).toBeCloseTo(mx, 10);
      expect(centers[c]?.[1]).toBeCloseTo(my, 10);
    }
  });

  it("predict labels by the closest leaf entry and validates input", () => {
    const birch = new Birch({ threshold: 0.4, branchingFactor: 3, nClusters: 3 }).fit(X);
    const probe = tensor(
      [
        [0, 0],
        [4, 0],
        [0, 4],
      ],
      { dtype: "float64" }
    );
    expect(canon(flat(birch.predict(probe)))).toEqual([0, 1, 2]);
    expect(flat(birch.predict(X))).toEqual(flat(birch.labels));
    expect(() => birch.predict(tensor([[1, 2, 3]], { dtype: "float64" }))).toThrow(ShapeError);
  });

  it("partialFit adds samples to the tree and partialFit() reclusters after setParams", () => {
    const birch = new Birch({ threshold: 0.4, branchingFactor: 3, nClusters: 3 });
    birch.partialFit(tensor(BI_X.slice(0, 15), { dtype: "float64" }));
    expect(birch.labels.size).toBe(15);
    birch.partialFit(tensor(BI_X.slice(15), { dtype: "float64" }));
    expect(birch.labels.size).toBe(15);
    expect(canon(flat(birch.predict(X)))).toEqual([
      ...new Array(10).fill(0),
      ...new Array(10).fill(1),
      ...new Array(10).fill(2),
    ]);

    birch.setParams({ nClusters: 2 });
    birch.partialFit();
    expect(birch.clusterCenters.shape[0]).toBe(2);

    expect(() => birch.partialFit(tensor([[1, 2, 3]], { dtype: "float64" }))).toThrow(ShapeError);
    expect(() => new Birch().partialFit()).toThrow(NotFittedError);
  });

  it("fit discards the previous tree", () => {
    const birch = new Birch({ threshold: 0.4, branchingFactor: 3, nClusters: 3 });
    birch.fit(X);
    birch.fit(
      tensor(
        [
          [100, 100, 100],
          [101, 100, 100],
          [0, 0, 0],
          [1, 0, 0],
        ],
        { dtype: "float64" }
      )
    );
    expect(birch.clusterCenters.shape[1]).toBe(3);
  });

  it("validates options including NaN and null", () => {
    expect(() => new Birch({ threshold: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new Birch({ threshold: Number.POSITIVE_INFINITY })).toThrow(InvalidParameterError);
    expect(() => new Birch({ nClusters: 0 })).toThrow(InvalidParameterError);
    expect(() => new Birch({ branchingFactor: 1.5 })).toThrow(InvalidParameterError);
    const birch = new Birch();
    expect(() => birch.setParams({ threshold: Number.NaN })).toThrow(InvalidParameterError);
    birch.setParams({ nClusters: null });
    expect(birch.getParams().nClusters).toBeNull();
    const other = new Birch();
    other.setParams(birch.getParams());
    expect(other.getParams()).toEqual(birch.getParams());
  });

  it("handles a single sample", () => {
    const birch = new Birch({ nClusters: 1 }).fit(tensor([[3, 4]], { dtype: "float64" }));
    expect(flat(birch.labels)).toEqual([0]);
    expect(flat(birch.clusterCenters)).toEqual([3, 4]);
  });
});
