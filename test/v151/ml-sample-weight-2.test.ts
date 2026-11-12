import { describe, expect, it } from "vitest";
import * as ml from "../../src/ml";
import { tensor } from "../../src/ndarray";

// Expected values: scikit-learn 1.9.1, fit(X, y, sample_weight=w), predict_proba rounded to 12 digits.
const REF = {
  MultinomialNB: [
    [0.731166641333, 0.062038467385, 0.206794891282],
    [0.138885733522, 0.856546021033, 0.004568245446],
    [0.003004859834, 0.133288120343, 0.863707019823],
    [0.965128757029, 0.025446582708, 0.009424660262],
    [0.022842865196, 0.854158334619, 0.122998800185],
    [0.033059598064, 0.050891600102, 0.916048801834],
    [0.973030278185, 0.009456915442, 0.017512806373],
    [0.016404557382, 0.974242712577, 0.009352730041],
  ],
  BernoulliNB: [
    [0.937952583558, 0.0271393839, 0.034908032542],
    [0.032534857733, 0.934090854447, 0.03337428782],
    [0.08233000889, 0.157582182536, 0.760087808574],
    [0.937952583558, 0.0271393839, 0.034908032542],
    [0.012911832781, 0.778479885423, 0.208608281797],
    [0.08233000889, 0.157582182536, 0.760087808574],
    [0.937952583558, 0.0271393839, 0.034908032542],
    [0.032534857733, 0.934090854447, 0.03337428782],
  ],
  ComplementNB: [
    [0.706498269252, 0.062810513702, 0.230691217046],
    [0.122152201105, 0.822337290188, 0.055510508707],
    [0.027332321365, 0.147676416075, 0.82499126256],
    [0.894781225764, 0.054485180824, 0.050733593412],
    [0.05399244431, 0.619981386439, 0.326026169251],
    [0.164997748748, 0.072574124001, 0.762428127251],
    [0.935595125906, 0.018360256239, 0.046044617856],
    [0.030962626738, 0.882737874006, 0.086299499256],
  ],
  CategoricalNB: [
    [0.616738661157, 0.189695006094, 0.193566332749],
    [0.082734079505, 0.858841339571, 0.058424580923],
    [0.361570851856, 0.314007811885, 0.324421336259],
    [0.938409766418, 0.038201537285, 0.023388696297],
    [0.057278555224, 0.845644865795, 0.097076578982],
    [0.089196016998, 0.123456141541, 0.787347841461],
    [0.728132542758, 0.168626650694, 0.103240806548],
    [0.037937043585, 0.933486828997, 0.028576127418],
  ],
  Quantile: [1.0, 0.8999999999999999],
};

const X = tensor(
  [
    [2, 0, 1],
    [1, 3, 0],
    [0, 1, 4],
    [3, 1, 0],
    [0, 2, 2],
    [1, 0, 3],
    [4, 1, 1],
    [0, 3, 1],
  ],
  { dtype: "float64" }
);
const y = tensor([0, 1, 2, 0, 1, 2, 0, 1], { dtype: "int32" });
const w = tensor([1, 2, 0.5, 3, 1, 1.5, 0.25, 2], { dtype: "float64" });

function expectClose(actual: number[][], expected: number[][], tol: number): void {
  expect(actual.length).toBe(expected.length);
  actual.forEach((row, i) => {
    row.forEach((v, j) => {
      expect(Math.abs(v - (expected[i]?.[j] as number))).toBeLessThan(tol);
    });
  });
}

describe("ML-14 naive Bayes sampleWeight matches scikit-learn", () => {
  const cases: [
    "MultinomialNB" | "BernoulliNB" | "ComplementNB" | "CategoricalNB",
    () => unknown,
  ][] = [
    ["MultinomialNB", () => new ml.MultinomialNB()],
    ["BernoulliNB", () => new ml.BernoulliNB({ binarize: 1.0 })],
    ["ComplementNB", () => new ml.ComplementNB()],
    ["CategoricalNB", () => new ml.CategoricalNB()],
  ];
  for (const [name, make] of cases) {
    it(name, () => {
      const model = make() as unknown as {
        fit(X: unknown, y: unknown, w: unknown): unknown;
        predictProba(X: unknown): { toArray(): unknown };
      };
      model.fit(X, y, w);
      expectClose(model.predictProba(X).toArray() as number[][], REF[name], 1e-9);
    });
  }
});

describe("ML-14 QuantileRegressor sampleWeight matches scikit-learn", () => {
  it("weighted fit gives the same coefficient and intercept", () => {
    const Xq = tensor([[0], [1], [2], [3], [4], [5], [6], [7]], { dtype: "float64" });
    const yq = tensor([0.5, 1.8, 2.2, 3.9, 4.1, 5.5, 6.2, 7.9], { dtype: "float64" });
    const q = new ml.QuantileRegressor({ quantile: 0.7, alpha: 0.05 }).fit(Xq, yq, w);
    const coef = Array.from(q.coef as ArrayLike<number>);
    expect(Math.abs((coef[0] as number) - REF.Quantile[0])).toBeLessThan(1e-6);
    expect(Math.abs(Number(q.intercept) - REF.Quantile[1])).toBeLessThan(1e-6);
  });
});

describe("ML-14 RANSAC, Bagging and Gaussian processes with sampleWeight", () => {
  const Xr = tensor([[0], [1], [2], [3], [4], [5], [6], [7]], { dtype: "float64" });
  const yr = tensor([0, 1.2, 1.9, 3.1, 4.2, 4.8, 6.3, 6.9], { dtype: "float64" });
  const wr = tensor([10, 10, 10, 10, 0.01, 0.01, 0.01, 0.01], { dtype: "float64" });

  it("RANSACRegressor passes weights to the final refit", () => {
    const plain = new ml.RANSACRegressor({ randomState: 0, residualThreshold: 5 }).fit(Xr, yr);
    const weighted = new ml.RANSACRegressor({ randomState: 0, residualThreshold: 5 }).fit(
      Xr,
      yr,
      wr
    );
    expect(weighted.predict(Xr).toArray()).not.toEqual(plain.predict(Xr).toArray());
  });

  it("BaggingRegressor never draws zero-weight rows", () => {
    const zero = tensor([1, 1, 1, 1, 0, 0, 0, 0], { dtype: "float64" });
    const model = new ml.BaggingRegressor({ randomState: 0, nEstimators: 5 }).fit(Xr, yr, zero);
    // Every member saw only the first four rows, whose largest target is 3.1.
    for (const v of model.predict(Xr).toArray() as number[]) expect(v).toBeLessThanOrEqual(3.1);
  });

  it("Gaussian processes reject sampleWeight", () => {
    expect(() => new ml.GaussianProcessRegressor().fit(Xr, yr, wr as never)).toThrow(
      /does not support sampleWeight/
    );
    expect(() =>
      new ml.GaussianProcessClassifier().fit(
        Xr,
        tensor([0, 0, 0, 0, 1, 1, 1, 1], { dtype: "int32" }),
        wr as never
      )
    ).toThrow(/does not support sampleWeight/);
  });
});
