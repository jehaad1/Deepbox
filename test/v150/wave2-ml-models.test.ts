import { describe, expect, it } from "vitest";
import {
  catchWarnings,
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
} from "../../src/core";
import {
  AdaBoostClassifier,
  BaggingClassifier,
  BayesianRidge,
  BernoulliNB,
  CategoricalNB,
  ComplementNB,
  crossValidate,
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  ElasticNet,
  ExtraTreesClassifier,
  GaussianNB,
  GradientBoostingClassifier,
  GradientBoostingRegressor,
  getEstimatorTags,
  HuberRegressor,
  KNeighborsClassifier,
  LinearRegression,
  LinearSVC,
  MultinomialNB,
  NearestNeighbors,
  OneVsOneClassifier,
  OneVsRestClassifier,
  QuantileRegressor,
  RandomForestClassifier,
  RandomForestRegressor,
  Ridge,
  SGDClassifier,
  SGDRegressor,
  StackingClassifier,
  SVC,
  VotingClassifier,
} from "../../src/ml";
import type {
  AdaBoostClassifierOptions,
  BaggingOptions,
  GradientBoostingLoss,
  StackingClassifierOptions,
  StackingMethod,
} from "../../src/ml/ensemble";
import { predictRegressionTree } from "../../src/ml/ensemble/_tree";
import type { ExtraTreesClassifierOptions, ExtraTreesOptions } from "../../src/ml/tree";
import { sortPositionsByColumn } from "../../src/ml/tree/DecisionTree";
import { type Tensor, tensor } from "../../src/ndarray";
import { setSeed } from "../../src/random";

// Compile-time check: the option and enum types are reachable from the module indexes.
type ExportedTypes = [
  AdaBoostClassifierOptions,
  BaggingOptions,
  ExtraTreesClassifierOptions,
  ExtraTreesOptions,
  GradientBoostingLoss,
  StackingClassifierOptions,
  StackingMethod,
];
const exportedTypesCheck: ExportedTypes | undefined = undefined;

function lcg(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807) % 2147483647;
    return s / 2147483647;
  };
}

function f64(data: number[] | number[][]): Tensor {
  return tensor(data, { dtype: "float64" });
}

/** Same data as the scikit-learn reference script (same LCG, same formulas). */
function referenceData(): { X: number[][]; y: number[]; yc: number[] } {
  const rnd = lcg(7);
  const X: number[][] = [];
  for (let i = 0; i < 200; i++) X.push([rnd() * 4 - 2, rnd() * 4 - 2, rnd() * 4 - 2]);
  const y = X.map((r) => 3 * r[0]! - 2 * r[1]! ** 2 + 0.5 * Math.sin(5 * r[2]!));
  const yc = X.map((r) => (r[0]! + r[1]! * r[2]! > 0.2 ? 1 : 0));
  return { X, y, yc };
}

function blobs(n = 90, seed = 3): { X: Tensor; y: Tensor } {
  const rnd = lcg(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(rnd())) * Math.cos(2 * Math.PI * rnd());
  const centers = [
    [0, 0],
    [3, 0],
    [0, 3],
  ];
  const rows: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = i % 3;
    rows.push([centers[c]![0]! + gauss() * 0.8, centers[c]![1]! + gauss() * 0.8]);
    labels.push(c);
  }
  return { X: f64(rows), y: tensor(Int32Array.from(labels), { dtype: "int32" }) };
}

function binaryData(n = 80, seed = 11): { X: Tensor; y: Tensor } {
  const rnd = lcg(seed);
  const rows: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < n; i++) {
    const a = rnd();
    const b = rnd();
    rows.push([a, b, rnd()]);
    labels.push(a + b > 1 ? 1 : 0);
  }
  return { X: f64(rows), y: f64(labels) };
}

function toNumbers(t: Tensor): number[] {
  return Array.from(t.data as ArrayLike<number>);
}

describe("random seeds (shared helper)", () => {
  it("exports the option types from the module indexes", () => {
    expect(exportedTypesCheck).toBeUndefined();
  });

  it("RandomForest gives different forests for fractional seeds 0.5 and 0.7", () => {
    const { X, y } = binaryData();
    const a = new RandomForestClassifier({ nEstimators: 8, randomState: 0.5 }).fit(X, y);
    const b = new RandomForestClassifier({ nEstimators: 8, randomState: 0.7 }).fit(X, y);
    const c = new RandomForestClassifier({ nEstimators: 8, randomState: 0.5 }).fit(X, y);
    expect(toNumbers(a.predictProba(X))).toEqual(toNumbers(c.predictProba(X)));
    expect(toNumbers(a.predictProba(X))).not.toEqual(toNumbers(b.predictProba(X)));
  });

  it("Bagging, AdaBoost, GradientBoosting and SGD tell fractional seeds apart", () => {
    const { X, y } = binaryData();
    const bag = (seed: number) =>
      toNumbers(
        new BaggingClassifier({ nEstimators: 6, maxSamples: 0.6, randomState: seed })
          .fit(X, y)
          .predictProba(X)
      );
    expect(bag(0.5)).not.toEqual(bag(0.7));
    expect(bag(0.5)).toEqual(bag(0.5));

    const ada = (seed: number) =>
      toNumbers(
        new AdaBoostClassifier({ nEstimators: 10, randomState: seed }).fit(X, y).decisionFunction(X)
      );
    expect(ada(0.5)).not.toEqual(ada(0.7));

    const gb = (seed: number) =>
      toNumbers(
        new GradientBoostingClassifier({
          nEstimators: 5,
          subsample: 0.5,
          randomState: seed,
        })
          .fit(X, y)
          .decisionFunction(X)
      );
    expect(gb(0.5)).not.toEqual(gb(0.7));

    const rows: number[][] = [];
    const target: number[] = [];
    const rnd = lcg(5);
    for (let i = 0; i < 60; i++) {
      const a = rnd();
      rows.push([a, rnd()]);
      target.push(2 * a + 1);
    }
    const sgd = (seed: number) =>
      Array.from(
        new SGDRegressor({ maxIter: 3, tol: 0, randomState: seed }).fit(f64(rows), f64(target)).coef
      );
    expect(sgd(0.5)).not.toEqual(sgd(0.7));
    expect(sgd(0.5)).toEqual(sgd(0.5));
  });

  it("negative and large seeds are valid and reproducible", () => {
    const { X, y } = binaryData();
    for (const seed of [-5, 2 ** 40]) {
      const a = new ExtraTreesClassifier({ nEstimators: 4, randomState: seed }).fit(X, y);
      const b = new ExtraTreesClassifier({ nEstimators: 4, randomState: seed }).fit(X, y);
      expect(toNumbers(a.predictProba(X))).toEqual(toNumbers(b.predictProba(X)));
    }
  });

  it("LinearSVC and ElasticNet (random selection) tell fractional seeds apart", () => {
    const { X, y } = binaryData(60, 23);
    const svc = (seed: number) =>
      toNumbers(new LinearSVC({ randomState: seed, maxIter: 3, tol: 0 }).fit(X, y).coef);
    expect(svc(0.2)).not.toEqual(svc(0.7));
    expect(svc(0.2)).toEqual(svc(0.2));
    const target = f64(toNumbers(y).map((v, i) => v * 2 + (i % 7) / 10));
    const enet = (seed: number) =>
      toNumbers(
        new ElasticNet({
          alpha: 0.001,
          selection: "random",
          randomState: seed,
          maxIter: 2,
          tol: 0,
        }).fit(X, target).coef
      );
    expect(enet(0.2)).not.toEqual(enet(0.7));
    expect(enet(0.2)).toEqual(enet(0.2));
  });

  it("the global seed makes maxFeatures sampling reproducible without randomState", () => {
    const { X, y } = binaryData();
    const run = () => {
      setSeed(21);
      return toNumbers(
        new RandomForestClassifier({ nEstimators: 4, maxFeatures: 1 }).fit(X, y).predictProba(X)
      );
    };
    expect(run()).toEqual(run());
    const tree = () => {
      setSeed(4);
      return toNumbers(new DecisionTreeClassifier({ maxFeatures: 1 }).fit(X, y).predictProba(X));
    };
    expect(tree()).toEqual(tree());
  });
});

describe("estimator tags and classes", () => {
  it("NearestNeighbors is unsupervised, not a classifier", () => {
    const tags = getEstimatorTags(new NearestNeighbors());
    expect(tags.estimatorType).toBe("transformer");
    expect(tags.requiresY).toBe(false);
    expect(getEstimatorTags(new KNeighborsClassifier()).estimatorType).toBe("classifier");
  });

  it("classifiers expose the label order through `classes`", () => {
    const { X, y } = blobs();
    for (const model of [
      new KNeighborsClassifier(),
      new GradientBoostingClassifier({ nEstimators: 3 }),
      new LinearSVC(),
    ]) {
      expect("classes" in model).toBe(true);
      model.fit(X, y);
      expect(toNumbers(model.classes as Tensor)).toEqual([0, 1, 2]);
    }
  });
});

describe("DecisionTree behaviour", () => {
  it("stops at pure nodes and matches scikit-learn on continuous float64 data", () => {
    const { X, y, yc } = referenceData();
    // scikit-learn: n_leaves, sum(predict), sum(predict ** 2) on the training data.
    const reg = [
      { depth: 3, leaves: 8, sq: 4225.428578583834 },
      { depth: 5, leaves: 31, sq: 4780.5599633608 },
      { depth: Number.POSITIVE_INFINITY, leaves: 200, sq: 4907.62751461617 },
    ];
    for (const c of reg) {
      const tree = new DecisionTreeRegressor({ maxDepth: c.depth }).fit(f64(X), f64(y));
      const pred = toNumbers(tree.predict(f64(X)));
      expect(tree.getNLeaves()).toBe(c.leaves);
      expect(pred.reduce((a, b) => a + b, 0)).toBeCloseTo(-479.26121415216363, 8);
      expect(pred.reduce((a, b) => a + b * b, 0)).toBeCloseTo(c.sq, 7);
    }
    const clf = [
      { depth: 3, leaves: 8 },
      { depth: 5, leaves: 18 },
      { depth: Number.POSITIVE_INFINITY, leaves: 26 },
    ];
    for (const c of clf) {
      const tree = new DecisionTreeClassifier({ maxDepth: c.depth }).fit(f64(X), f64(yc));
      expect(tree.getNLeaves()).toBe(c.leaves);
    }
  });

  it("does not split pure nodes, so constant runs use no depth", () => {
    const x = f64(Array.from({ length: 12 }, (_, i) => [i]));
    const step = new DecisionTreeRegressor({ maxDepth: 5 }).fit(
      x,
      f64([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1])
    );
    expect(step.getNLeaves()).toBe(2);
    expect(step.getDepth()).toBe(1);
    const steps = new DecisionTreeRegressor({ maxDepth: Number.POSITIVE_INFINITY }).fit(
      x,
      f64([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    );
    expect(steps.getNLeaves()).toBe(3);
    expect(steps.getDepth()).toBe(2);
  });

  it("sorts float64 columns that are not float32-exact like a comparator sort", () => {
    const rnd = lcg(9);
    for (const n of [95, 96, 97, 500, 4000]) {
      for (let mode = 0; mode < 6; mode++) {
        const col = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          let v = (rnd() - 0.5) * (mode === 1 ? 1e-300 : mode === 2 ? 1e300 : 100);
          if (mode === 3) v = Math.round(v * 0.1) / 3 + 0; // many ties, no negative zero
          if (mode === 4) v = i * 1.1 + 0.3; // already ascending
          if (mode === 5) v = (n - i) * 1.1 + 0.3; // descending
          col[i] = v;
        }
        const positions = new Int32Array(n);
        for (let i = 0; i < n; i++) positions[i] = i;
        sortPositionsByColumn(col, positions, n);
        const expected = Array.from({ length: n }, (_, i) => i).sort(
          (a, b) => (col[a] as number) - (col[b] as number)
        );
        expect(Array.from(positions)).toEqual(expected);
      }
    }
  });

  it("fits a long sorted float64 column with the same tree as float64 data in other orders", () => {
    const rnd = lcg(13);
    const n = 300;
    const rows: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < n; i++) {
      const a = rnd() * 7.123456789;
      rows.push([a, rnd()]);
      y.push(Math.sin(a) + 0.01 * i);
    }
    const order = Array.from({ length: n }, (_, i) => i).sort(
      (p, q) => (rows[p]![0] as number) - (rows[q]![0] as number)
    );
    const a = new DecisionTreeRegressor({ maxDepth: 6 }).fit(f64(rows), f64(y));
    const b = new DecisionTreeRegressor({ maxDepth: 6 }).fit(
      f64(order.map((i) => rows[i]!)),
      f64(order.map((i) => y[i]!))
    );
    expect(a.getNLeaves()).toBe(b.getNLeaves());
    const probe = f64(rows.slice(0, 40));
    const pa = toNumbers(a.predict(probe));
    const pb = toNumbers(b.predict(probe));
    pa.forEach((v, i) => {
      expect(v).toBeCloseTo(pb[i] as number, 10);
    });
  });
});

describe("RandomForest maxFeatures null", () => {
  it("null examines every feature, like a single decision tree", () => {
    const { X, y } = binaryData(120, 17);
    const forest = new RandomForestClassifier({
      nEstimators: 1,
      bootstrap: false,
      maxFeatures: null,
      maxDepth: Number.POSITIVE_INFINITY,
    }).fit(X, y);
    const tree = new DecisionTreeClassifier({ maxDepth: Number.POSITIVE_INFINITY }).fit(X, y);
    expect(toNumbers(forest.predict(X))).toEqual(toNumbers(tree.predict(X)));
    expect(forest.getParams()["maxFeatures"]).toBeNull();
    expect(new RandomForestRegressor({ maxFeatures: null }).getParams()["maxFeatures"]).toBeNull();
    expect(() => new RandomForestClassifier({ maxFeatures: 0 })).toThrow(InvalidParameterError);
  });
});

describe("ensembles", () => {
  it("flattens a very deep tree without overflowing the stack", () => {
    const depth = 120000;
    type Node = {
      isLeaf: boolean;
      prediction?: number;
      featureIndex?: number;
      threshold?: number;
      left?: Node;
      right?: Node;
    };
    const root: Node = { isLeaf: false, featureIndex: 0, threshold: 0.5 };
    let node = root;
    for (let i = 0; i < depth; i++) {
      node.left = { isLeaf: true, prediction: i };
      const next: Node = { isLeaf: false, featureIndex: 0, threshold: 1 + i };
      node.right = next;
      node = next;
    }
    node.left = { isLeaf: true, prediction: -1 };
    node.right = { isLeaf: true, prediction: -2 };
    const fake = { tree_: root } as unknown as DecisionTreeRegressor;
    const out = predictRegressionTree(fake, Float64Array.from([0, 0.75, 3.5]), 3, 1);
    expect(Array.from(out)).toEqual([0, 1, 4]);
  });

  it("Stacking keeps its own copy of the estimator list", () => {
    const { X, y } = binaryData(60, 29);
    const first = new DecisionTreeClassifier({ maxDepth: 2 });
    const late = new DecisionTreeClassifier({ maxDepth: 2 });
    const list = [first];
    const model = new StackingClassifier({ estimators: list, cv: 3 });
    list.push(late);
    model.fit(X, y);
    expect(() => late.predict(X)).toThrow(NotFittedError);
    expect(first.predict(X).size).toBe(60);
    expect(model.clone().getParams()["nEstimators"]).toBe(1);
  });

  it("Bagging, AdaBoost and Stacking refuse fractional labels instead of truncating them", () => {
    const { X } = binaryData(40, 31);
    const y = f64(Array.from({ length: 40 }, (_, i) => (i % 2 === 0 ? 0.5 : 2.5)));
    expect(() => new BaggingClassifier({ nEstimators: 2 }).fit(X, y)).toThrow(DataValidationError);
    expect(() => new AdaBoostClassifier({ nEstimators: 2 }).fit(X, y)).toThrow(DataValidationError);
    expect(() =>
      new StackingClassifier({ estimators: [new DecisionTreeClassifier()] }).fit(X, y)
    ).toThrow(DataValidationError);
  });

  it("VotingClassifier keeps fractional labels and float64 probabilities", () => {
    const { X } = binaryData(50, 37);
    const y = f64(Array.from({ length: 50 }, (_, i) => (i % 2 === 0 ? 0.5 : 2.5)));
    const model = new VotingClassifier({
      estimators: [new DecisionTreeClassifier({ maxDepth: 3 }), new GaussianNB()],
      voting: "soft",
    }).fit(X, y);
    expect(new Set(toNumbers(model.predict(X)))).toEqual(new Set([0.5, 2.5]));
    expect(model.predictProba(X).dtype).toBe("float64");
  });

  it("matches scikit-learn for gradient boosting with the lad loss on pure gradients", () => {
    const n = 100;
    const x = f64(Array.from({ length: n }, (_, i) => [i]));
    const y = f64(
      Array.from({ length: n }, (_, i) => 1000 + 2000 * Math.sin(i / 8) + (1000 * i) / 10)
    );
    // scikit-learn 1.8: GradientBoostingRegressor(loss="absolute_error", max_depth=3,
    //   n_estimators=20, learning_rate=0.1, random_state=0).fit(x, y).predict(x)
    const model = new GradientBoostingRegressor({
      loss: "absolute_error",
      maxDepth: 3,
      nEstimators: 20,
      learningRate: 0.1,
      randomState: 0,
    }).fit(x, y);
    const pred = toNumbers(model.predict(x));
    expect(pred[0]).toBeCloseTo(2959.974280403932, 6);
    expect(pred[30]).toBeCloseTo(3378.965554005907, 6);
    expect(pred[60]).toBeCloseTo(8338.048326015445, 6);
    expect(pred[99]).toBeCloseTo(8334.372938902789, 6);
    expect(pred.reduce((a, p) => a + p, 0)).toBeCloseTo(587423.2316537689, 4);
    expect(model.score(x, y)).toBeCloseTo(0.9448737940232325, 8);
  });
});

describe("linear models", () => {
  const rows: number[][] = [];
  const target: number[] = [];
  const rnd = lcg(41);
  for (let i = 0; i < 40; i++) {
    const a = rnd() * 3;
    const b = rnd() * 3;
    rows.push([a, b]);
    target.push(1.23456789123 * a - 2.3456789123 * b + 0.12345678912 + 1e-3 * rnd());
  }
  const X = f64(rows);
  const y = f64(target);

  it("every linear model returns float64 predictions and coefficients", () => {
    const models = [
      new LinearRegression(),
      new Ridge({ alpha: 0.1 }),
      new ElasticNet({ alpha: 0.001 }),
      new HuberRegressor(),
      new BayesianRidge(),
      new QuantileRegressor({ alpha: 0 }),
      new SGDRegressor({ randomState: 1 }),
    ];
    for (const model of models) {
      model.fit(X, y);
      expect(model.predict(X).dtype).toBe("float64");
    }
    const exact = new Ridge({ alpha: 0 }).fit(
      f64([[0], [1], [2], [3]]),
      f64([1e-10, 1 + 1e-10, 2 + 1e-10, 3 + 1e-10])
    );
    expect(toNumbers(exact.predict(f64([[4]])))[0]).toBeCloseTo(4 + 1e-10, 12);
  });

  it("offers one coefficient and intercept type across the family", () => {
    const huber = new HuberRegressor().fit(X, y);
    const bayes = new BayesianRidge().fit(X, y);
    for (const model of [huber, bayes]) {
      expect(model.coefTensor.dtype).toBe("float64");
      expect(toNumbers(model.coefTensor)).toEqual(Array.from(model.coef));
    }
    const ols = new LinearRegression().fit(X, y);
    expect(typeof ols.interceptValue).toBe("number");
    expect(ols.interceptValue).toBeCloseTo(Number(ols.intercept?.data[0]), 14);
    expect(new LinearRegression({ fitIntercept: false }).fit(X, y).interceptValue).toBe(0);
    expect(() => new LinearRegression().interceptValue).toThrow(NotFittedError);
  });

  it("score shares one R2 implementation and validates targets the same way", () => {
    const models = [
      new Ridge({ alpha: 0.1 }),
      new SGDRegressor({ randomState: 1 }),
      new QuantileRegressor({ alpha: 0 }),
      new LinearRegression(),
    ];
    for (const model of models) {
      model.fit(X, y);
      expect(model.score(X, y)).toBeLessThanOrEqual(1);
      expect(() => model.score(X, f64([]))).toThrow(DataValidationError);
    }
    const ridge = new Ridge({ alpha: 0 }).fit(X, y);
    const pred = toNumbers(ridge.predict(X));
    const truth = target;
    const mean = truth.reduce((a, b) => a + b, 0) / truth.length;
    const ssRes = truth.reduce((a, t, i) => a + (t - (pred[i] as number)) ** 2, 0);
    const ssTot = truth.reduce((a, t) => a + (t - mean) ** 2, 0);
    expect(ridge.score(X, y)).toBeCloseTo(1 - ssRes / ssTot, 12);
    expect(() => new Ridge().score(X, y)).toThrow(NotFittedError);
  });

  it("Ridge with solver svd returns the minimum-norm solution for collinear columns", () => {
    // scikit-learn: Ridge(alpha=0).fit(X, y) gives coef [0.23, 0.46], intercept -0.25.
    const collinear = f64([
      [1, 2],
      [2, 4],
      [3, 6],
      [4, 8],
    ]);
    const target4 = f64([1, 2, 3, 4.5]);
    const model = new Ridge({ alpha: 0, solver: "svd" }).fit(collinear, target4);
    const coef = toNumbers(model.coef);
    expect(coef[0]).toBeCloseTo(0.23, 10);
    expect(coef[1]).toBeCloseTo(0.46, 10);
    expect(Number(model.intercept)).toBeCloseTo(-0.25, 10);
  });

  it("Ridge warns when an iterative solver runs out of iterations", () => {
    for (const solver of ["lsqr", "sag"] as const) {
      const warnings = catchWarnings(() => {
        new Ridge({ alpha: 0.001, solver, maxIter: 1, tol: 1e-14 }).fit(X, y);
      });
      expect(warnings.length).toBe(1);
      expect(warnings[0]?.category).toBe("ConvergenceWarning");
      expect(warnings[0]?.source).toBe("Ridge");
    }
    const quiet = catchWarnings(() => {
      new Ridge({ alpha: 0.1, solver: "lsqr", maxIter: 500, tol: 1e-8 }).fit(X, y);
      new Ridge({ alpha: 0.1, solver: "auto" }).fit(X, y);
    });
    expect(quiet.length).toBe(0);
  });
});

describe("SGDClassifier coefficients", () => {
  it("coefTensor has the (n_classes, n_features) layout of the other classifiers", () => {
    const { X, y } = blobs();
    const model = new SGDClassifier({ randomState: 1 }).fit(X, y);
    expect(model.coefTensor.shape).toEqual([3, 2]);
    expect(model.coef.length).toBe(6);
    expect(Array.from(model.coefTensor.data as ArrayLike<number>)).toEqual(Array.from(model.coef));
  });
});

describe("naive Bayes clone", () => {
  it("clones every variant with its hyperparameters and no fitted state", () => {
    const counts = f64([
      [1, 0, 3],
      [2, 1, 0],
      [0, 4, 1],
      [3, 0, 0],
    ]);
    const labels = f64([0, 1, 1, 0]);
    const models = [
      new MultinomialNB({ alpha: 0.5, classPrior: [0.3, 0.7] }),
      new ComplementNB({ alpha: 0.25 }),
      new BernoulliNB({ alpha: 2 }),
      new CategoricalNB({ alpha: 0.1, minCategories: 6 }),
      new GaussianNB({ varSmoothing: 1e-6, priors: [0.4, 0.6] }),
    ];
    for (const model of models) {
      model.fit(counts, labels);
      const copy = model.clone();
      expect(copy).not.toBe(model);
      expect(copy.constructor).toBe(model.constructor);
      expect(copy.getParams()).toEqual(model.getParams());
      expect(copy.classes).toBeUndefined();
      expect(() => copy.predict(counts)).toThrow(NotFittedError);
    }
  });
});

describe("SVM and meta-estimators", () => {
  it("SVC predicts and scores multiclass int labels, also inside crossValidate", () => {
    const { X, y } = blobs();
    const svc = new SVC().fit(X, y);
    expect(svc.predict(X).dtype).toBe("int32");
    expect(svc.score(X, y)).toBeGreaterThan(0.9);
    expect(svc.decisionFunction(X).shape).toEqual([90, 3]);
    const result = crossValidate(new SVC(), X, y, { cv: 3 });
    expect(result.testScores["score"]?.length).toBe(3);
    for (const s of result.testScores["score"] ?? []) expect(s).toBeGreaterThan(0.8);
  });

  it("LinearSVC works under one-vs-rest and one-vs-one through its decisionFunction", () => {
    const { X, y } = blobs();
    expect(typeof new LinearSVC().decisionFunction).toBe("function");
    const ovr = new OneVsRestClassifier({ estimator: new LinearSVC() }).fit(X, y);
    const ovo = new OneVsOneClassifier({ estimator: new LinearSVC() }).fit(X, y);
    expect(ovr.score(X, y)).toBeGreaterThan(0.9);
    expect(ovo.score(X, y)).toBeGreaterThan(0.9);
    const svcOvr = new OneVsRestClassifier({ estimator: new SVC() }).fit(X, y);
    expect(svcOvr.score(X, y)).toBeGreaterThan(0.9);
  });
});
