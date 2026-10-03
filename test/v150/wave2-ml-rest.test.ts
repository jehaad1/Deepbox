/**
 * Wave 2 regression tests for the shared parts of `deepbox/ml` (group ml-rest):
 * integer targets in `MLPRegressor.score`, estimator tag inference, `TSNE.setParams`,
 * the shared estimator cloning used by the meta-estimators, the `deepbox/ml` barrel and the
 * internal helpers moved to `src/ml/_internal.ts`.
 */

import { describe, expect, it } from "vitest";
import { DataValidationError, DTypeError, InvalidParameterError } from "../../src/core";
import {
  assertEstimator,
  type BayesianRidgeOptions,
  CalibratedClassifierCV,
  type ClassWeightOption,
  type DecisionTreeClassifierOptions,
  type ElasticNetOptions,
  type ExtraTreesOptions,
  type ForestMaxFeatures,
  GaussianMixture,
  GaussianProcessRegressor,
  GridSearchCV,
  getEstimatorTags,
  type HuberRegressorOptions,
  IsolationForest,
  Isomap,
  type IsotonicOutOfBounds,
  type IsotonicRegressionOptions,
  johnsonLindenstraussMinDim,
  type KernelRidgeKernel,
  type KernelRidgeOptions,
  type KernelType,
  KMeans,
  type LassoOptions,
  LinearSVC,
  type LinearSVCOptions,
  LocalOutlierFactor,
  LogisticRegression,
  MeanShift,
  MiniBatchKMeans,
  MLPRegressor,
  OneClassSVM,
  OneVsRestClassifier,
  OPTICS,
  PCA,
  Pipeline,
  type RandomForestOptions,
  Ridge,
  SelfTrainingClassifier,
  type SelfTrainingTermination,
  SpectralClustering,
  type SVCOptions,
  type TreeMaxFeatures,
  TSNE,
  type TSNEOptions,
} from "../../src/ml";
import { digamma, jacobiEigenSymmetric, kthSmallest, r2Score } from "../../src/ml/_internal";
import { toFloat64View } from "../../src/ml/_validation";
import { jacobiEigenSymmetric as jacobiFromDecomposition } from "../../src/ml/decomposition";
import { Tensor, tensor, transpose } from "../../src/ndarray";
import { StandardScaler } from "../../src/preprocess";

/** Three well separated blobs of 20 points each in 3 dimensions. */
function blobs(dtype: "float64" | "int64" | "int32" = "float64"): { X: Tensor; y: Tensor } {
  const rows: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < 60; i++) {
    const c = i % 3;
    rows.push([c * 30 + ((i * 7) % 11), c * -20 + ((i * 5) % 13) + 40, ((i * 3) % 7) + c * 10]);
    labels.push(c);
  }
  return { X: tensor(rows, { dtype }), y: tensor(labels, { dtype }) };
}

describe("MLPRegressor.score with integer targets", () => {
  it("accepts int64 targets (BigInt data) and matches the float64 score", () => {
    const f = blobs("float64");
    const i = blobs("int64");
    const options = { hiddenLayerSizes: [4], maxIter: 5, randomState: 1 };
    const scoreFloat = new MLPRegressor(options).fit(f.X, f.y).score(f.X, f.y);
    const model = new MLPRegressor(options).fit(i.X, i.y);
    expect(model.score(i.X, i.y)).toBeCloseTo(scoreFloat, 10);
  });

  it("still rejects non-finite targets and non-numeric dtypes", () => {
    const { X, y } = blobs();
    const model = new MLPRegressor({ hiddenLayerSizes: [3], maxIter: 2, randomState: 1 }).fit(X, y);
    const bad = tensor(
      Array.from({ length: 60 }, (_, i) => (i === 3 ? Number.NaN : i)),
      {
        dtype: "float64",
      }
    );
    expect(() => model.score(X, bad)).toThrow(DataValidationError);
    const strings = tensor(Array.from({ length: 60 }, () => "a"));
    expect(() => model.score(X, strings)).toThrow(DTypeError);
  });
});

describe("getEstimatorTags inference order", () => {
  const fn = (): unknown => undefined;

  it("treats a mixture-like object (scoreSamples + fitPredict + predictProba) as a clusterer", () => {
    const mixture = {
      fit: fn,
      getParams: fn,
      setParams: fn,
      scoreSamples: fn,
      fitPredict: fn,
      predict: fn,
      predictProba: fn,
    };
    const tags = getEstimatorTags(mixture as never);
    expect(tags.estimatorType).toBe("clusterer");
    expect(tags.requiresY).toBe(false);
    expect(tags.hasPredictProba).toBe(true);
  });

  it("keeps outlier detectors (scoreSamples without predictProba) as outlier detectors", () => {
    const detector = {
      fit: fn,
      getParams: fn,
      setParams: fn,
      scoreSamples: fn,
      fitPredict: fn,
      predict: fn,
    };
    expect(getEstimatorTags(detector as never).estimatorType).toBe("outlier_detector");
  });

  it("pins the type of the built-in estimators that cross-validation relies on", () => {
    expect(getEstimatorTags(new GaussianMixture({ nComponents: 2 })).estimatorType).toBe(
      "clusterer"
    );
    expect(getEstimatorTags(new KMeans({ nClusters: 2 })).estimatorType).toBe("clusterer");
    expect(getEstimatorTags(new IsolationForest()).estimatorType).toBe("outlier_detector");
    expect(getEstimatorTags(new LocalOutlierFactor()).estimatorType).toBe("outlier_detector");
    expect(getEstimatorTags(new OneClassSVM()).estimatorType).toBe("outlier_detector");
    expect(getEstimatorTags(new LogisticRegression()).estimatorType).toBe("classifier");
    expect(getEstimatorTags(new LinearSVC()).estimatorType).toBe("classifier");
    expect(getEstimatorTags(new Ridge()).estimatorType).toBe("regressor");
    expect(getEstimatorTags(new GaussianProcessRegressor()).estimatorType).toBe("regressor");
    expect(getEstimatorTags(new PCA({ nComponents: 1 })).estimatorType).toBe("transformer");
    expect(
      getEstimatorTags(
        new Pipeline([
          ["s", new StandardScaler()],
          ["r", new Ridge()],
        ])
      ).estimatorType
    ).toBe("regressor");
    expect(
      getEstimatorTags(new CalibratedClassifierCV({ estimator: new LogisticRegression() }))
        .estimatorType
    ).toBe("classifier");
  });
});

describe("TSNE.setParams", () => {
  const { X } = blobs();

  it("is a complete estimator (assertEstimator accepts it) and returns this", () => {
    const model = new TSNE({ perplexity: 5, nIter: 50, randomState: 0 });
    expect(assertEstimator(model)).toBe(model);
    expect(model.setParams({ nIter: 60 })).toBe(model);
    expect(model.getParams()["nIter"]).toBe(60);
  });

  it("changes the fit and re-derives defaults that depend on perplexity", () => {
    const model = new TSNE({ perplexity: 5, nIter: 40, randomState: 0 });
    expect(model.getParams()["approximateNeighbors"]).toBe(15);
    model.setParams({ perplexity: 8 });
    expect(model.getParams()["perplexity"]).toBe(8);
    expect(model.getParams()["approximateNeighbors"]).toBe(24);
    expect(model.getParams()["negativeSamples"]).toBe(16);

    const explicit = new TSNE({ perplexity: 5, approximateNeighbors: 9 });
    explicit.setParams({ perplexity: 8 });
    expect(explicit.getParams()["approximateNeighbors"]).toBe(9);

    const a = new TSNE({ perplexity: 8, nIter: 40, randomState: 0 }).fitTransform(X);
    const b = model.setParams({ nIter: 40, randomState: 0 }).fitTransform(X);
    expect(Array.from(b.data as Float64Array)).toEqual(Array.from(a.data as Float64Array));
  });

  it("keeps the fitted embedding intact until the next fit", () => {
    const model = new TSNE({ perplexity: 8, nIter: 30, randomState: 0 });
    model.fit(X);
    model.setParams({ nComponents: 3 });
    expect(model.embedding.shape).toEqual([60, 2]);
    expect(model.transform().shape).toEqual([60, 2]);
    expect(model.fitTransform(X).shape).toEqual([60, 3]);
  });

  it("rejects unknown and invalid parameters without changing the estimator", () => {
    const model = new TSNE({ perplexity: 8, nIter: 30 });
    expect(() => model.setParams({ bogus: 1 })).toThrow(InvalidParameterError);
    expect(() => model.setParams({ perplexity: -1 })).toThrow(InvalidParameterError);
    expect(() => model.setParams({ method: "fast" })).toThrow(InvalidParameterError);
    expect(model.getParams()["perplexity"]).toBe(8);
    expect(model.getParams()["method"]).toBe("exact");
  });

  it("round-trips through getParams (clone by constructor)", () => {
    const model = new TSNE({
      perplexity: 7,
      nIter: 33,
      earlyExaggerationIter: 100,
      randomState: 3,
    });
    const copy = new TSNE(model.getParams() as TSNEOptions);
    expect(copy.getParams()).toEqual(model.getParams());
  });
});

describe("shared estimator cloning in meta-estimators", () => {
  /** Needs a constructor argument and has no clone(), so it can be neither cloned nor rebuilt. */
  class Rigid {
    fitCalls = 0;
    constructor(required: { a: number }) {
      if (typeof required.a !== "number") throw new Error("a is required");
    }
    fit(): this {
      this.fitCalls++;
      return this;
    }
    predict(X: Tensor): Tensor {
      return tensor(new Float64Array(X.shape[0] ?? 0));
    }
    predictProba(X: Tensor): Tensor {
      const n = X.shape[0] ?? 0;
      return tensor(new Float64Array(n * 2).fill(0.5)).reshape([n, 2]);
    }
    score(): number {
      return 0;
    }
    getParams(): Record<string, unknown> {
      return {};
    }
    setParams(): this {
      return this;
    }
  }

  it("SelfTrainingClassifier no longer fits an estimator it cannot clone in place", () => {
    const base = new Rigid({ a: 1 });
    const model = new SelfTrainingClassifier({ baseEstimator: base as never });
    const X = tensor([[0], [1], [2], [3]], { dtype: "float64" });
    const y = tensor([0, 1, -1, -1], { dtype: "float64" });
    expect(() => model.fit(X, y)).toThrow(InvalidParameterError);
    expect(() => model.fit(X, y)).toThrow(/cannot clone/);
    expect(base.fitCalls).toBe(0);
    expect(() => model.clone()).toThrow(InvalidParameterError);
  });

  it("SelfTrainingClassifier fits a clone and leaves the caller's estimator unfitted", () => {
    const { X, y } = blobs();
    const labels = Array.from({ length: 60 }, (_, i) => (i % 3 === 0 ? (i % 9 === 0 ? 0 : 1) : -1));
    const base = new LogisticRegression();
    const model = new SelfTrainingClassifier({ baseEstimator: base });
    model.fit(X, tensor(labels, { dtype: "float64" }));
    expect(() => base.predict(X)).toThrow();
    expect(model.predict(X).size).toBe(60);
    expect(y.size).toBe(60);
  });

  it("CalibratedClassifierCV and OneVsRestClassifier report an estimator they cannot clone", () => {
    const { X, y } = blobs();
    const calibrated = new CalibratedClassifierCV({
      estimator: new Rigid({ a: 1 }) as never,
      cv: 3,
    });
    expect(() => calibrated.fit(X, y)).toThrow(InvalidParameterError);
    const ovr = new OneVsRestClassifier({ estimator: new Rigid({ a: 1 }) as never });
    expect(() => ovr.fit(X, y)).toThrow(/cannot clone/);
  });

  it("GridSearchCV tunes a Pipeline through step__param keys", () => {
    const rows: number[][] = [];
    const labels: number[] = [];
    for (let i = 0; i < 60; i++) {
      const c = i % 2;
      rows.push([c * 3 + Math.sin(i), Math.cos(i * 2) + c]);
      labels.push(c);
    }
    const pipe = new Pipeline([
      ["scaler", new StandardScaler()],
      ["clf", new LogisticRegression()],
    ]);
    const search = new GridSearchCV(
      pipe,
      { clf__maxIter: [5, 50], scaler__withMean: [true, false] },
      { cv: 3 }
    );
    search.fit(tensor(rows, { dtype: "float64" }), tensor(labels, { dtype: "float64" }));
    expect(Object.keys(search.bestParams).sort()).toEqual(["clf__maxIter", "scaler__withMean"]);
  });
});

describe("float64 outputs of the clustering estimators", () => {
  it("never return float32 tensors for float64 input", () => {
    const { X } = blobs();
    const models = [
      new MiniBatchKMeans({ nClusters: 3, randomState: 0 }),
      new MeanShift(),
      new OPTICS({ minSamples: 5 }),
      new SpectralClustering({ nClusters: 3, randomState: 0 }),
    ];
    for (const model of models) {
      model.fit(X);
      const record = model as unknown as Record<string, unknown>;
      for (const key of ["clusterCenters", "labels", "predictProba", "transform"]) {
        if (!(key in record)) continue;
        const value =
          typeof record[key] === "function"
            ? (record[key] as (x: Tensor) => unknown)(X)
            : record[key];
        if (value instanceof Tensor) expect(value.dtype).not.toBe("float32");
      }
    }
    expect(new MiniBatchKMeans({ nClusters: 3, randomState: 0 }).fit(X).clusterCenters.dtype).toBe(
      "float64"
    );
    expect(new MeanShift().fit(X).clusterCenters.dtype).toBe("float64");
  });

  it("fits integer (BigInt) data like float data", () => {
    const f = blobs("float64");
    const i = blobs("int64");
    const a = new KMeans({ nClusters: 3, randomState: 1 }).fit(f.X);
    const b = new KMeans({ nClusters: 3, randomState: 1 }).fit(i.X);
    expect(Array.from(b.labels.data as Int32Array)).toEqual(
      Array.from(a.labels.data as Int32Array)
    );
    expect(() => new Isomap({ nNeighbors: 40, nComponents: 2 }).fitTransform(i.X)).not.toThrow();
  });
});

describe("deepbox/ml barrel", () => {
  it("exports johnsonLindenstraussMinDim", () => {
    expect(johnsonLindenstraussMinDim(1e6, 0.5)).toBe(663);
  });

  it("exports the option and helper types of the estimator modules", () => {
    // These are compile-time checks: tsc fails on this file when a type is missing from the barrel.
    const termination: SelfTrainingTermination = "max_iter";
    const kernel: KernelRidgeKernel = "rbf";
    const bounds: IsotonicOutOfBounds = "clip";
    const kernelType: KernelType = "rbf";
    const weights: ClassWeightOption = "balanced";
    const maxFeatures: ForestMaxFeatures = "sqrt";
    const treeFeatures: TreeMaxFeatures = "log2";
    const lasso: LassoOptions = {};
    const elastic: ElasticNetOptions = {};
    const bayes: BayesianRidgeOptions = {};
    const huber: HuberRegressorOptions = {};
    const isotonic: IsotonicRegressionOptions = {};
    const kernelRidge: KernelRidgeOptions = {};
    const svc: SVCOptions = {};
    const linearSvc: LinearSVCOptions = {};
    const forest: RandomForestOptions = {};
    const extra: ExtraTreesOptions = {};
    const tree: DecisionTreeClassifierOptions = {};
    const tsne: TSNEOptions = {};
    expect([
      termination,
      kernel,
      bounds,
      kernelType,
      weights,
      maxFeatures,
      treeFeatures,
    ]).toHaveLength(7);
    expect([
      lasso,
      elastic,
      bayes,
      huber,
      isotonic,
      kernelRidge,
      svc,
      linearSvc,
      forest,
      extra,
      tree,
      tsne,
    ]).toHaveLength(12);
  });
});

describe("src/ml/_internal helpers", () => {
  it("kthSmallest returns the rank-k value for every k", () => {
    const values = [5, 1, 4, 4, 9, -2, 7, 0, 4, 3];
    const sorted = [...values].sort((a, b) => a - b);
    for (let k = 0; k < values.length; k++) {
      expect(kthSmallest(Float64Array.from(values), k)).toBe(sorted[k]);
    }
  });

  it("r2Score follows the scikit-learn conventions", () => {
    expect(r2Score([1, 2, 3, 4], [1, 2, 3, 4])).toBe(1);
    expect(r2Score([1, 2, 3, 4], [2.5, 2.5, 2.5, 2.5])).toBeCloseTo(0, 12);
    expect(r2Score([3, 3, 3], [3, 3, 3])).toBe(1);
    expect(r2Score([3, 3, 3], [3, 3, 4])).toBe(0);
    // sklearn.metrics.r2_score([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]) = 0.9486081370449679
    expect(r2Score([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])).toBeCloseTo(0.9486081370449679, 12);
  });

  it("keeps jacobiEigenSymmetric reachable from the decomposition module", () => {
    expect(jacobiFromDecomposition).toBe(jacobiEigenSymmetric);
    const { values } = jacobiEigenSymmetric(Float64Array.from([2, 1, 1, 2]), 2);
    expect(values[0]).toBeCloseTo(3, 12);
    expect(values[1]).toBeCloseTo(1, 12);
    // scipy.special.digamma(1.0) = -0.5772156649015329
    expect(digamma(1)).toBeCloseTo(-0.5772156649015329, 11);
  });

  it("toFloat64View names the argument in its error messages", () => {
    const wide = transpose(
      tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { dtype: "float64" }
      )
    );
    expect(() => toFloat64View(wide, "weights")).toThrow(/weights must be contiguous/);
    expect(() => toFloat64View(tensor(["a", "b"]), "labels")).toThrow(
      /labels must have a real numeric dtype/
    );
    expect(() => toFloat64View(tensor(["a", "b"]))).toThrow(
      /tensor must have a real numeric dtype/
    );
  });
});
