import { describe, expect, it } from "vitest";
import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  NotImplementedError,
  ShapeError,
} from "../../src/core/errors";
import { tensor, transpose, zeros } from "../../src/ndarray";
import {
  MaxAbsScaler,
  MinMaxScaler,
  Normalizer,
  PowerTransformer,
  QuantileTransformer,
  RobustScaler,
  StandardScaler,
} from "../../src/preprocess/scalers";
import { SplineTransformer } from "../../src/preprocess/spline";

const f64 = (rows: number[][]) => tensor(rows, { dtype: "float64" });
const col = (values: number[]) => f64(values.map((v) => [v]));
const flat = (t: { toArray(): unknown }): number[] => (t.toArray() as number[][]).flat();

function expectClose(actual: number[], expected: number[], tol = 1e-12): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs((actual[i] as number) - (expected[i] as number))).toBeLessThanOrEqual(tol);
  }
}

describe("c54 StandardScaler", () => {
  it("centers a constant feature exactly (mean is clamped into [min, max])", () => {
    // 0.1 + 0.1 + 0.1 = 0.30000000000000004; /3 used to give a mean above 0.1.
    const out = new StandardScaler().fitTransform(col([0.1, 0.1, 0.1]));
    expect(flat(out)).toEqual([0, 0, 0]);
  });

  it("does not snap tiny-valued data to zero on inverseTransform", () => {
    const scaler = new StandardScaler().fit(col([1e-13, 3e-13, 5e-13]));
    // reference: 3e-13 + sqrt(8/3) * 1e-13 = 4.632993161855452e-13
    const inv = flat(scaler.inverseTransform(col([0, 1])));
    expect(inv[0]).toBeCloseTo(3e-13, 25);
    expect(inv[1]).toBeCloseTo(4.632993161855452e-13, 25);
    expect(inv[1]).not.toBe(5e-13);
  });

  it("does not snap near-integer data to the integer", () => {
    const scaler = new StandardScaler().fit(col([4, 5.0000000001, 6]));
    const z = scaler.transform(col([5.0000000001]));
    expect(flat(scaler.inverseTransform(z))[0]).toBeCloseTo(5.0000000001, 12);
    expect(flat(scaler.inverseTransform(z))[0]).not.toBe(5);
  });

  it("rejects transform input with a different number of features", () => {
    const scaler = new StandardScaler().fit(
      f64([
        [1, 2],
        [3, 4],
      ])
    );
    expect(() => scaler.transform(f64([[1, 2, 3]]))).toThrow(ShapeError);
    expect(() => scaler.inverseTransform(f64([[1]]))).toThrow(ShapeError);
  });

  it("matches scikit-learn standardisation (ddof=0)", () => {
    // sklearn.preprocessing.StandardScaler().fit_transform([[1,10],[2,20],[4,60]])
    const out = flat(
      new StandardScaler().fitTransform(
        f64([
          [1, 10],
          [2, 20],
          [4, 60],
        ])
      )
    );
    expectClose(
      out,
      [
        -1.0690449676496978, -0.9258200997725515, -0.2672612419124245, -0.46291004988627577,
        1.3363062095621219, 1.3887301496588271,
      ],
      1e-14
    );
  });

  it("partialFit over batches equals fit over all rows", () => {
    const a = new StandardScaler()
      .partialFit(
        f64([
          [1, 2],
          [3, 4],
        ])
      )
      .partialFit(f64([[10, 20]]))
      .partialFit(
        f64([
          [-4, 0],
          [7, 7],
        ])
      );
    const b = new StandardScaler().fit(
      f64([
        [1, 2],
        [3, 4],
        [10, 20],
        [-4, 0],
        [7, 7],
      ])
    );
    expect(a.nSamplesSeen).toBe(5);
    expectClose(a.mean?.toArray() as number[], b.mean?.toArray() as number[], 1e-12);
    expectClose(a.variance?.toArray() as number[], b.variance?.toArray() as number[], 1e-12);
    expect(() => a.partialFit(f64([[1, 2, 3]]))).toThrow(ShapeError);
  });

  it("fit discards earlier state", () => {
    const s = new StandardScaler().fit(
      f64([
        [1, 2],
        [3, 4],
      ])
    );
    s.fit(col([5, 6, 7]));
    expect(s.nFeaturesIn).toBe(1);
    expect(s.nSamplesSeen).toBe(3);
    expect(s.mean?.toArray()).toEqual([6]);
  });

  it("exposes fitted attributes and honours options", () => {
    const s = new StandardScaler({ withMean: false }).fit(col([1, 3, 5]));
    expect(s.mean).toBeUndefined();
    expect(s.scale?.toArray()).toEqual([Math.sqrt(8 / 3)]);
    expect(new StandardScaler().mean).toBeUndefined();
  });

  it("treats a zero-variance feature as unscaled", () => {
    const out = new StandardScaler().fitTransform(
      f64([
        [1, 5],
        [2, 5],
        [3, 5],
      ])
    );
    expect(flat(out)).toEqual([-1.224744871391589, 0, 0, 0, 1.224744871391589, 0]);
  });

  it("validates options in setParams and keeps statistics", () => {
    const s = new StandardScaler().fit(col([1, 2, 3]));
    expect(() => s.setParams({ withMean: "yes" })).toThrow(InvalidParameterError);
    s.setParams({ withMean: false });
    expect(flat(s.transform(col([2])))[0]).toBeCloseTo(2 / Math.sqrt(2 / 3), 12);
    expect(s.clone().getParams()).toEqual({ withMean: false, withStd: true });
  });

  it("rejects data whose variance cannot be represented and keeps the previous fit", () => {
    const s = new StandardScaler().fit(col([1, 2, 3]));
    expect(() => s.fit(f64([[1e200], [-1e200]]))).toThrow(DataValidationError);
    expect(() => s.partialFit(f64([[1e200], [-1e200]]))).toThrow(DataValidationError);
    expect(s.mean?.toArray()).toEqual([2]);
    expect(s.nSamplesSeen).toBe(3);
  });

  it("rejects an empty feature axis with a feature message", () => {
    expect(() => new StandardScaler().fit(zeros([3, 0]))).toThrow(/at least one feature/);
  });

  it("handles strided input and zero-row transform", () => {
    const s = new StandardScaler().fit(
      transpose(
        f64([
          [1, 2, 3],
          [4, 5, 6],
        ])
      )
    );
    expect(s.transform(zeros([0, 2])).shape).toEqual([0, 2]);
  });
});

describe("c54 MinMaxScaler", () => {
  it("shifts unseen values of a constant feature like scikit-learn", () => {
    // sklearn: MinMaxScaler().fit([[2],[2]]).transform([[2],[5]]) -> [[0],[3]]
    const s = new MinMaxScaler().fit(col([2, 2]));
    expect(flat(s.transform(col([2, 5])))).toEqual([0, 3]);
    expect(flat(s.inverseTransform(col([0, 3])))).toEqual([2, 5]);
  });

  it("maps the fitted extremes exactly onto the feature range", () => {
    const s = new MinMaxScaler({ featureRange: [0.1, 0.7] });
    const out = flat(s.fitTransform(col([3.3, 7.1, 11.9, 1e6])));
    expect(out[0]).toBe(0.1);
    expect(out[3]).toBe(0.7);
  });

  it("matches scikit-learn on a custom range", () => {
    // MinMaxScaler((-2, 3)).fit_transform([[1, 10], [2, 40], [4, 20]])
    const out = flat(
      new MinMaxScaler({ featureRange: [-2, 3] }).fitTransform(
        f64([
          [1, 10],
          [2, 40],
          [4, 20],
        ])
      )
    );
    expectClose(out, [-2, -2, -0.3333333333333335, 3, 3, -0.3333333333333335], 1e-14);
  });

  it("partialFit tracks running min and max", () => {
    const s = new MinMaxScaler().partialFit(col([2, 4])).partialFit(col([0, 3]));
    expect(s.dataMin?.toArray()).toEqual([0]);
    expect(s.dataMax?.toArray()).toEqual([4]);
    expect(s.dataRange?.toArray()).toEqual([4]);
    expect(s.scale?.toArray()).toEqual([0.25]);
    expect(s.nSamplesSeen).toBe(4);
  });

  it("rejects mismatched feature counts and bad ranges", () => {
    const s = new MinMaxScaler().fit(
      f64([
        [1, 2],
        [3, 4],
      ])
    );
    expect(() => s.transform(col([1, 2]))).toThrow(ShapeError);
    expect(() => new MinMaxScaler({ featureRange: [0, Number.NaN] })).toThrow(
      InvalidParameterError
    );
    expect(() => s.setParams({ featureRange: [2, 1] })).toThrow(InvalidParameterError);
  });

  it("clone and getParams round trip the options", () => {
    const s = new MinMaxScaler({ featureRange: [-1, 1], clip: true });
    const c = new MinMaxScaler(s.getParams() as { featureRange: [number, number]; clip: boolean });
    expect(c.getParams()).toEqual({ featureRange: [-1, 1], clip: true });
    expect(s.clone().getParams()).toEqual(s.getParams());
  });
});

describe("c54 MaxAbsScaler", () => {
  it("divides exactly so the largest magnitude maps to exactly +-1", () => {
    // 49 * (1 / 49) === 0.9999999999999999 in IEEE doubles, 49 / 49 === 1.
    const out = flat(new MaxAbsScaler().fitTransform(col([49, -49, 1])));
    expect(out[0]).toBe(1);
    expect(out[1]).toBe(-1);
  });

  it("partialFit and getters", () => {
    const s = new MaxAbsScaler().partialFit(f64([[1, -2]])).partialFit(f64([[-3, 1]]));
    expect(s.maxAbs?.toArray()).toEqual([3, 2]);
    expect(s.nSamplesSeen).toBe(2);
    expect(() => s.partialFit(col([1]))).toThrow(ShapeError);
  });

  it("an all-zero feature is left unchanged in both directions", () => {
    const s = new MaxAbsScaler().fit(
      f64([
        [0, 1],
        [0, 2],
      ])
    );
    expect(s.scale?.toArray()).toEqual([1, 2]);
    expect(flat(s.inverseTransform(f64([[5, 1]])))).toEqual([5, 2]);
  });
});

describe("c54 RobustScaler", () => {
  it("matches scikit-learn with unitVariance (sklearn: [-1.3489795003921634, ..., 1.3489795003921634])", () => {
    const out = flat(new RobustScaler({ unitVariance: true }).fitTransform(col([0, 1, 2, 3, 4])));
    expectClose(
      out,
      [-1.3489795003921634, -0.6744897501960817, 0, 0.6744897501960817, 1.3489795003921634],
      1e-14
    );
  });

  it("unitVariance needs a quantile range strictly inside (0, 100)", () => {
    expect(() => new RobustScaler({ unitVariance: true, quantileRange: [0, 100] })).toThrow(
      /strictly between 0 and 100/
    );
    expect(() =>
      new RobustScaler().setParams({ unitVariance: true, quantileRange: [0, 90] })
    ).toThrow(InvalidParameterError);
  });

  it("a constant feature is centered and left unscaled", () => {
    expect(flat(new RobustScaler().fitTransform(col([0.1, 0.1, 0.1, 0.1])))).toEqual([0, 0, 0, 0]);
  });

  it("rejects mismatched feature counts and works after toggling options", () => {
    const s = new RobustScaler().fit(
      f64([
        [1, 2],
        [3, 4],
        [5, 9],
      ])
    );
    expect(() => s.transform(col([1, 2]))).toThrow(ShapeError);
    s.setParams({ withCentering: false });
    expect(flat(s.transform(f64([[2, 4]])))).toEqual([1, 4 / 3.5]);
    expect(s.center).toBeUndefined();
    expect(s.scale?.toArray()).toEqual([2, 3.5]);
  });

  it("changing the quantile range discards the fit", () => {
    const s = new RobustScaler().fit(col([1, 2, 3, 4]));
    s.setParams({ quantileRange: [10, 90] });
    expect(() => s.transform(col([1]))).toThrow(NotFittedError);
  });
});

describe("c54 Normalizer", () => {
  it("does not overflow or underflow for extreme row magnitudes", () => {
    const out = new Normalizer()
      .transform(
        f64([
          [1e200, 1e200],
          [1e-200, 1e-200],
          [0, 0],
        ])
      )
      .toArray();
    expectClose(
      (out as number[][]).flat(),
      [Math.SQRT1_2, Math.SQRT1_2, Math.SQRT1_2, Math.SQRT1_2, 0, 0],
      2e-16
    );
  });

  it("l1 and l2 norms that overflow still give unit-norm rows", () => {
    const l1 = flat(new Normalizer({ norm: "l1" }).transform(f64([[1.5e308, 1.5e308]])));
    expect(l1).toEqual([0.5, 0.5]);
    const l2 = flat(new Normalizer().transform(f64([[1.7e308, 1.7e308]])));
    expectClose(l2, [Math.SQRT1_2, Math.SQRT1_2], 2e-16);
  });

  it("scales the largest entry to exactly 1 for the max norm", () => {
    const out = flat(
      new Normalizer({ norm: "max" }).transform(
        f64([
          [49, 1],
          [-3, 3],
        ])
      )
    );
    expect(out).toEqual([1, 1 / 49, -1, 1]);
  });

  it("fit validates input and params can be changed", () => {
    expect(() => new Normalizer().fit(col([1]).reshape([1]))).toThrow();
    const n = new Normalizer().setParams({ norm: "l1" });
    expect(n.getParams()).toEqual({ norm: "l1" });
    expect(() => n.setParams({ norm: "l3" })).toThrow(InvalidParameterError);
    expect(flat(n.fitTransform(f64([[1, 3]])))).toEqual([0.25, 0.75]);
  });
});

describe("c54 QuantileTransformer", () => {
  it("normal output matches scikit-learn to double precision", () => {
    // QuantileTransformer(output_distribution="normal").fit_transform([[1],[2],[3],[4],[5]])
    const out = flat(
      new QuantileTransformer({ outputDistribution: "normal" }).fitTransform(col([1, 2, 3, 4, 5]))
    );
    expectClose(
      out,
      [-5.199337582605575, -0.6744897501960817, 0, 0.6744897501960817, 5.19933758270342],
      2e-10
    );
    expect(out[1]).toBeCloseTo(-0.6744897501960817, 14);
  });

  it("uniform output matches scikit-learn", () => {
    // QuantileTransformer(n_quantiles=3).fit_transform([[1],[2],[3],[4],[10]])
    const out = flat(
      new QuantileTransformer({ nQuantiles: 3 }).fitTransform(col([1, 2, 3, 4, 10]))
    );
    expectClose(out, [0, 0.25, 0.5, 0.5714285714285714, 1], 1e-15);
  });

  it("maps tied values to the average of the quantiles they cover", () => {
    // quantiles [1,2,2,2,9] at probabilities [0, .25, .5, .75, 1]
    const qt = new QuantileTransformer({ nQuantiles: 5 }).fit(col([1, 2, 2, 2, 9]));
    const out = flat(qt.transform(col([1, 2, 5, 9, 20, 0])));
    expectClose(out, [0, 0.5, 0.8571428571428571, 1, 1, 0], 1e-15);
  });

  it("normal inverse round trip is accurate away from the clipped tails", () => {
    const qt = new QuantileTransformer({ outputDistribution: "normal", nQuantiles: 5 });
    const x = col([1, 2, 3, 4, 5]);
    const back = flat(qt.inverseTransform(qt.fitTransform(x)));
    expectClose(back.slice(1, 4), [2, 3, 4], 1e-12);
  });

  it("inverse maps the clipped ends of the output range to the training min and max", () => {
    // sklearn: inverse_transform(transform([[0], [9], [1], [5]])) -> [1, 5, 1, 5]
    const qt = new QuantileTransformer({ outputDistribution: "normal", nQuantiles: 5 }).fit(
      col([1, 2, 3, 4, 5])
    );
    expect(flat(qt.inverseTransform(qt.transform(col([0, 9, 1, 5]))))).toEqual([1, 5, 1, 5]);
    const uniform = new QuantileTransformer({ nQuantiles: 5 }).fit(col([1, 2, 3, 4, 5]));
    // The uniform output is not clipped, so there is no snapping (sklearn: 1.0000002, 4.9999998, 2.2).
    expectClose(
      flat(uniform.inverseTransform(col([5e-8, 1 - 5e-8, 0.3]))),
      [1.0000002, 4.9999998, 2.2],
      1e-12
    );
  });

  it("constant feature maps to the lower end (as scikit-learn)", () => {
    const qt = new QuantileTransformer({ nQuantiles: 3 }).fit(col([5, 5, 5]));
    expect(flat(qt.transform(col([4, 5, 6])))).toEqual([0, 0, 1]);
  });

  it("rejects mismatched feature counts", () => {
    const qt = new QuantileTransformer().fit(
      f64([
        [1, 2],
        [3, 4],
      ])
    );
    expect(() => qt.transform(col([1, 2]))).toThrow(ShapeError);
    expect(() => qt.inverseTransform(col([1, 2]))).toThrow(ShapeError);
  });

  it("exposes quantiles and references", () => {
    const qt = new QuantileTransformer({ nQuantiles: 3 }).fit(
      f64([
        [1, 10],
        [2, 20],
        [3, 60],
      ])
    );
    expect(qt.quantiles?.toArray()).toEqual([
      [1, 10],
      [2, 20],
      [3, 60],
    ]);
    expect(qt.references?.toArray()).toEqual([0, 0.5, 1]);
  });

  it("subsampling is reproducible and getParams/clone round trip", () => {
    const x = col([1, 2, 3, 4, 5, 6, 7, 8]);
    const a = new QuantileTransformer({ nQuantiles: 4, subsample: 4, randomState: 3 }).fit(x);
    const b = a.clone().fit(x);
    expect(a.quantiles?.toArray()).toEqual(b.quantiles?.toArray());
    expect(a.getParams()).toEqual({
      nQuantiles: 4,
      outputDistribution: "uniform",
      subsample: 4,
      randomState: 3,
    });
  });

  it("inverseTransform rejects NaN", () => {
    const qt = new QuantileTransformer().fit(col([1, 2, 3]));
    expect(() => qt.inverseTransform(col([Number.NaN]))).toThrow(DataValidationError);
  });
});

describe("c54 PowerTransformer", () => {
  const sample = [1, 2, 5, 9, 30];

  it("yeo-johnson matches scikit-learn", () => {
    // PowerTransformer(standardize=False).fit_transform(...): lambda -0.28481047267408793
    const pt = new PowerTransformer().fit(col(sample));
    expect((pt.lambdas?.toArray() as number[])[0]).toBeCloseTo(-0.28481047267408793, 6);
    expectClose(
      flat(pt.transform(col(sample))),
      [
        0.6290167679846722, 0.943343970298828, 1.4033601348927147, 1.688749368514109,
        2.1907582226354436,
      ],
      1e-6
    );
  });

  it("box-cox matches scikit-learn", () => {
    const pt = new PowerTransformer({ method: "box-cox" }).fit(col(sample));
    expect((pt.lambdas?.toArray() as number[])[0]).toBeCloseTo(-0.10428706176309988, 6);
  });

  it("standardised yeo-johnson on mixed-sign data matches scikit-learn", () => {
    const out = flat(
      new PowerTransformer({ standardize: true }).fitTransform(col([-3, -1, 0, 2, 8]))
    );
    expectClose(
      out,
      [
        -1.5090428987296156, -0.44589113845606243, -0.04408900867559529, 0.4876618317068174,
        1.5113612141544557,
      ],
      1e-6
    );
  });

  it("finds lambdas beyond 5 for tightly clustered data far from zero", () => {
    // scikit-learn (scipy) gives about 102 here; the likelihood keeps rising up to the overflow limit.
    const pt = new PowerTransformer({ method: "box-cox" }).fit(
      col([990, 995, 998, 999, 1000, 1000.5, 1001, 1001.2])
    );
    const lambda = (pt.lambdas?.toArray() as number[])[0] as number;
    expect(lambda).toBeGreaterThan(90);
    expect(lambda).toBeLessThan(110);
    expect(flat(pt.transform(col([1000]))).every(Number.isFinite)).toBe(true);
  });

  it("box-cox lambda is invariant to a change of scale", () => {
    const a = new PowerTransformer({ method: "box-cox" }).fit(col(sample));
    const b = new PowerTransformer({ method: "box-cox" }).fit(col(sample.map((v) => v * 1e-9)));
    expect((a.lambdas?.toArray() as number[])[0]).toBeCloseTo(
      (b.lambdas?.toArray() as number[])[0] as number,
      6
    );
  });

  it("inverse recovers the input, with and without standardisation", () => {
    for (const method of ["box-cox", "yeo-johnson"] as const) {
      for (const standardize of [false, true]) {
        const pt = new PowerTransformer({ method, standardize });
        const back = flat(pt.inverseTransform(pt.fitTransform(col(sample))));
        expectClose(back, sample, 1e-9);
      }
    }
  });

  it("rejects mismatched feature counts instead of silently ignoring columns", () => {
    const pt = new PowerTransformer().fit(
      f64([
        [1, 2],
        [3, 4],
        [5, 9],
      ])
    );
    expect(() => pt.transform(col([1, 2, 3]))).toThrow(ShapeError);
    expect(() => pt.inverseTransform(col([1, 2, 3]))).toThrow(ShapeError);
  });

  it("box-cox rejects non-positive transform input", () => {
    const pt = new PowerTransformer({ method: "box-cox" }).fit(col(sample));
    expect(() => pt.transform(col([0]))).toThrow(InvalidParameterError);
  });

  it("setParams toggles standardize without refitting but method change discards the fit", () => {
    const pt = new PowerTransformer().fit(col(sample));
    pt.setParams({ standardize: true });
    const z = flat(pt.transform(col(sample)));
    expect(z.reduce((s, v) => s + v, 0) / z.length).toBeCloseTo(0, 12);
    pt.setParams({ method: "box-cox" });
    expect(() => pt.transform(col(sample))).toThrow(NotFittedError);
    expect(pt.clone().getParams()).toEqual({ method: "box-cox", standardize: true });
  });

  it("standardize treats a constant feature as unscaled (no rounding-noise scale)", () => {
    // 0.1 * 3 / 3 is not exactly 0.1; the old scale was about 1.4e-17 and blew values up.
    const pt = new PowerTransformer({ standardize: true }).fit(col([0.1, 0.1, 0.1]));
    expect(pt.scale?.toArray()).toEqual([1]);
    expect(flat(pt.transform(col([0.1, 0.2])))).toEqual([0, 0.1]);
  });

  it("constant features keep lambda 1 and pass values through exactly", () => {
    const pt = new PowerTransformer().fit(col([3, 3, 3]));
    expect(pt.lambdas?.toArray()).toEqual([1]);
    expect(flat(pt.transform(col([2, -2.5, 0.1])))).toEqual([2, -2.5, 0.1]);
    expect(flat(pt.inverseTransform(col([2, -2.5, 0.1])))).toEqual([2, -2.5, 0.1]);
    const bc = new PowerTransformer({ method: "box-cox" }).fit(col([3, 3, 3]));
    expect(flat(bc.transform(col([2, 0.5])))).toEqual([1, -0.5]);
    expect(flat(bc.inverseTransform(col([1, -0.5])))).toEqual([2, 0.5]);
    expect(() => bc.inverseTransform(col([-1]))).toThrow(InvalidParameterError);
  });
});

describe("c54 SplineTransformer", () => {
  const X = col([0, 1, 2, 3, 4]);
  const probe = col([-1, 0.5, 2, 5]);
  // scipy.interpolate.BSpline on the clamped knots [0,0,0,2,4,4,4], degree 2.
  const CONSTANT = [
    [1, 0, 0, 0],
    [0.5625, 0.40625, 0.03125, 0],
    [0, 0.5, 0.5, 0],
    [0, 0, 0, 1],
  ];
  const CONTINUE = [
    [2.25, -1.375, 0.125, 0],
    [0.5625, 0.40625, 0.03125, 0],
    [0, 0.5, 0.5, 0],
    [0, 0.125, -1.375, 2.25],
  ];
  const LINEAR = [
    [2, -1, 0, 0],
    [0.5625, 0.40625, 0.03125, 0],
    [0, 0.5, 0.5, 0],
    [0, 0, -1, 2],
  ];

  for (const [mode, expected] of [
    ["constant", CONSTANT],
    ["continue", CONTINUE],
    ["linear", LINEAR],
  ] as const) {
    it(`extrapolation "${mode}" matches scipy BSpline`, () => {
      const st = new SplineTransformer({
        nKnots: 3,
        degree: 2,
        knots: "uniform",
        extrapolation: mode,
      }).fit(X);
      expectClose(flat(st.transform(probe)), expected.flat(), 1e-14);
    });
  }

  it("linear extrapolation is continuous with the in-range basis", () => {
    const st = new SplineTransformer({ nKnots: 4, degree: 3, extrapolation: "linear" }).fit(
      col([0, 1, 2, 3, 4, 5, 6, 7])
    );
    const at = flat(st.transform(col([7])));
    const near = flat(st.transform(col([7 + 1e-9])));
    expectClose(near, at, 1e-8);
  });

  it("periodic splines match scikit-learn", () => {
    // SplineTransformer(n_knots=4, degree=2, knots="uniform", extrapolation="periodic")
    const input = col([0, 0.5, 5, -1]);
    const train = col([0, 1, 2, 3, 4]);
    const withBias = new SplineTransformer({
      nKnots: 4,
      degree: 2,
      knots: "uniform",
      extrapolation: "periodic",
    }).fit(train);
    expect(withBias.nFeaturesOut).toBe(3);
    expectClose(
      flat(withBias.transform(input)),
      [
        0.5, 0.5, 0, 0.19531249999999997, 0.734375, 0.0703125, 0.03125, 0.6875, 0.28125, 0.6875,
        0.03125, 0.28125,
      ],
      1e-14
    );
    // Without the bias column a periodic basis drops its last function.
    const noBias = new SplineTransformer({
      nKnots: 4,
      degree: 2,
      knots: "uniform",
      extrapolation: "periodic",
      includeBias: false,
    }).fit(train);
    expect(noBias.nFeaturesOut).toBe(2);
    expectClose(
      flat(noBias.transform(input)),
      [0.5, 0.5, 0.19531249999999997, 0.734375, 0.03125, 0.6875, 0.6875, 0.03125],
      1e-14
    );
  });

  it("quantile knots are interpolated like numpy.percentile", () => {
    // np.percentile([0, 1, 2, 10], [0, 50, 100]) -> [0, 1.5, 10]
    const st = new SplineTransformer({ nKnots: 3, degree: 1 }).fit(col([0, 1, 2, 10]));
    expect(st.knotPositions).toEqual([[0, 1.5, 10]]);
  });

  it("rows sum to one inside and outside the range for constant/periodic extrapolation", () => {
    for (const extrapolation of ["constant", "periodic"] as const) {
      const st = new SplineTransformer({ nKnots: 5, degree: 3, extrapolation }).fit(
        col([0, 1, 2, 3, 4, 5, 6])
      );
      const out = st.transform(col([-3, 0, 1.3, 6, 11])).toArray() as number[][];
      for (const row of out) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    }
  });

  it("extrapolation 'error' rejects values outside the fitted range", () => {
    const st = new SplineTransformer({ extrapolation: "error" }).fit(X);
    expect(() => st.transform(col([5]))).toThrow(DataValidationError);
    expect(st.transform(col([4])).shape).toEqual([1, 7]);
  });

  it("gives all-zero columns for a constant training feature, like scikit-learn", () => {
    const st = new SplineTransformer({ nKnots: 3, degree: 2 }).fit(
      f64([
        [2, 0],
        [2, 1],
        [2, 2],
      ])
    );
    const out = st
      .transform(
        f64([
          [2, 1],
          [7, 0],
        ])
      )
      .toArray() as number[][];
    expect(out[0]?.slice(0, 4)).toEqual([0, 0, 0, 0]);
    expect(out[1]?.slice(0, 4)).toEqual([0, 0, 0, 0]);
    expect((out[0] as number[]).slice(4).some((v) => v > 0)).toBe(true);
    const strict = new SplineTransformer({ extrapolation: "error" }).fit(col([2, 2, 2]));
    expect(strict.transform(col([2])).toArray()).toEqual([[0, 0, 0, 0, 0, 0, 0]]);
    expect(() => strict.transform(col([3]))).toThrow(DataValidationError);
  });

  it("validates fit and transform input", () => {
    const st = new SplineTransformer();
    expect(() => st.fit(zeros([0, 1]))).toThrow(InvalidParameterError);
    expect(() => st.fit(col([0, Number.NaN, 2]))).toThrow(DataValidationError);
    st.fit(X);
    expect(() => st.transform(col([Number.NaN]))).toThrow(DataValidationError);
    expect(() => st.transform(f64([[1, 2]]))).toThrow(InvalidParameterError);
  });

  it("validates options", () => {
    expect(() => new SplineTransformer({ extrapolation: "wrap" as never })).toThrow(
      InvalidParameterError
    );
    expect(() => new SplineTransformer({ knots: "random" as never })).toThrow(
      InvalidParameterError
    );
    expect(
      () => new SplineTransformer({ extrapolation: "periodic", nKnots: 3, degree: 3 })
    ).toThrow(/degree < nKnots/);
  });

  it("inverseTransform throws NotImplementedError", () => {
    expect(() => new SplineTransformer().inverseTransform(X)).toThrow(NotImplementedError);
  });

  it("setParams applies valid options, rejects invalid ones atomically and discards the fit", () => {
    const st = new SplineTransformer().fit(X);
    expect(() => st.setParams({ degree: -1 })).toThrow(InvalidParameterError);
    expect(st.getParams()["degree"]).toBe(3);
    expect(st.nFeaturesOut).toBe(7);
    st.setParams({ nKnots: 4 });
    expect(() => st.transform(X)).toThrow(NotFittedError);
    expect(st.getParams()).toEqual({
      nKnots: 4,
      degree: 3,
      extrapolation: "constant",
      includeBias: true,
      knots: "quantile",
    });
    expect(st.clone().getParams()).toEqual(st.getParams());
  });

  it("returns float64 output for strided input", () => {
    const st = new SplineTransformer({ nKnots: 3, degree: 2 }).fit(
      transpose(f64([[0, 1, 2, 3, 4]]))
    );
    const out = st.transform(transpose(f64([[0, 2, 4]])));
    expect(out.dtype).toBe("float64");
    expect(out.shape).toEqual([3, 4]);
  });
});
