import { describe, expect, it } from "vitest";
import { DTypeError, InvalidParameterError, ShapeError } from "../../src/core";
import {
  explainedVarianceScore,
  mae,
  meanAbsolutePercentageError,
  medianAbsoluteError,
  mse,
  r2Score,
  rmse,
} from "../../src/metrics";
import { tensor as makeTensor } from "../../src/ndarray";

const tensor = (data: number[] | number[][] | string[][]) =>
  makeTensor(
    data,
    typeof data[0] === "object" && typeof (data[0] as unknown[])[0] === "string"
      ? {}
      : { dtype: "float64" }
  );

// Expected values come from scikit-learn 1.9.1: mean_squared_error, root_mean_squared_error,
// mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error and
// median_absolute_error with multioutput="uniform_average" | "raw_values" | [weights] |
// "variance_weighted" and with sample_weight.

type Reference = {
  yt: number[][];
  yp: number[][];
  yt2: number[][];
  yp2: number[][];
  sw: number[];
  raw1d: number[];
  res: { fn: string; mo: string | number[]; w: boolean; val: number | number[] }[];
};

const REF: Reference = {
  yt: [
    [5.072, -1.398, 0.01],
    [1.223, -2.367, 0.001],
    [-0.003, -5.264, 0.305],
    [1.801, -1.876, -0.052],
    [1.516, -0.784, -0.073],
    [-4.36, 1.664, 0.037],
    [0.823, -4.58, 0.495],
  ],
  yp: [
    [5.226, -1.785, 0.213],
    [1.178, -3.818, -0.04],
    [-2.291, -4.215, 0.264],
    [1.058, -0.804, -0.217],
    [2.051, -2.848, -0.139],
    [-5.564, 3.126, 0.214],
    [0.494, -3.739, 0.477],
  ],
  sw: [1, 2, 0.5, 3, 1, 1, 2],
  res: [
    { fn: "mse", mo: "uniform_average", w: false, val: 0.9225808095238094 },
    { fn: "mse", mo: "uniform_average", w: true, val: 0.7170651746031748 },
    {
      fn: "mse",
      mo: "raw_values",
      w: false,
      val: [1.0938308571428568, 1.6585108571428573, 0.015400714285714285],
    },
    {
      fn: "mse",
      mo: "raw_values",
      w: true,
      val: [0.595591238095238, 1.5400405238095243, 0.015563761904761907],
    },
    { fn: "mse", mo: [0.5, 0.3, 0.2], w: false, val: 1.0475488285714283 },
    { fn: "mse", mo: [0.5, 0.3, 0.2], w: true, val: 0.7629205285714287 },
    { fn: "rmse", mo: "uniform_average", w: false, val: 0.819265049143671 },
    { fn: "rmse", mo: "uniform_average", w: true, val: 0.712494692581664 },
    {
      fn: "rmse",
      mo: "raw_values",
      w: false,
      val: [1.045863689561339, 1.287831843503979, 0.12409961436569529],
    },
    {
      fn: "rmse",
      mo: "raw_values",
      w: true,
      val: [0.771745578604269, 1.240983691999828, 0.124754807140895],
    },
    { fn: "rmse", mo: [0.5, 0.3, 0.2], w: false, val: 0.9341013207050023 },
    { fn: "rmse", mo: [0.5, 0.3, 0.2], w: true, val: 0.7831188583302618 },
    { fn: "mae", mo: "uniform_average", w: false, val: 0.6826190476190476 },
    { fn: "mae", mo: "uniform_average", w: true, val: 0.6136825396825397 },
    {
      fn: "mae",
      mo: "raw_values",
      w: false,
      val: [0.7568571428571428, 1.1894285714285715, 0.10157142857142858],
    },
    {
      fn: "mae",
      mo: "raw_values",
      w: true,
      val: [0.5727619047619047, 1.1654761904761903, 0.10280952380952382],
    },
    { fn: "mae", mo: [0.5, 0.3, 0.2], w: false, val: 0.7555714285714286 },
    { fn: "mae", mo: [0.5, 0.3, 0.2], w: true, val: 0.6565857142857142 },
    { fn: "r2", mo: "uniform_average", w: false, val: 0.6957783332267158 },
    { fn: "r2", mo: "uniform_average", w: true, val: 0.6802593142001269 },
    {
      fn: "r2",
      mo: "raw_values",
      w: false,
      val: [0.837719779338407, 0.6446827074709696, 0.6049325128707708],
    },
    {
      fn: "r2",
      mo: "raw_values",
      w: true,
      val: [0.8697818651390616, 0.5237670353986539, 0.6472290420626651],
    },
    { fn: "r2", mo: [0.5, 0.3, 0.2], w: false, val: 0.7332512044846484 },
    { fn: "r2", mo: [0.5, 0.3, 0.2], w: true, val: 0.72146685160166 },
    { fn: "r2", mo: "variance_weighted", w: false, val: 0.7582135873004879 },
    { fn: "r2", mo: "variance_weighted", w: true, val: 0.7260220905498275 },
    { fn: "ev", mo: "uniform_average", w: false, val: 0.7121029598382885 },
    { fn: "ev", mo: "uniform_average", w: true, val: 0.7036550042185863 },
    {
      fn: "ev",
      mo: "raw_values",
      w: false,
      val: [0.8842453247343429, 0.6458740672973932, 0.6061894874831295],
    },
    {
      fn: "ev",
      mo: "raw_values",
      w: true,
      val: [0.9124036298147233, 0.5303457751901683, 0.6682156076508673],
    },
    { fn: "ev", mo: [0.5, 0.3, 0.2], w: false, val: 0.7571227800530154 },
    { fn: "ev", mo: [0.5, 0.3, 0.2], w: true, val: 0.7489486689945856 },
    { fn: "ev", mo: "variance_weighted", w: false, val: 0.786099352727571 },
    { fn: "ev", mo: "variance_weighted", w: true, val: 0.7536776502700899 },
    { fn: "mape", mo: "uniform_average", w: false, val: 39.99344596969102 },
    { fn: "mape", mo: "uniform_average", w: true, val: 16.15701079716928 },
    {
      fn: "mape",
      mo: "raw_values",
      w: false,
      val: [109.16788270629623, 0.7650608939520154, 10.047394308824805],
    },
    {
      fn: "mape",
      mo: "raw_values",
      w: true,
      val: [36.58128519357913, 0.6852648617823318, 11.204482336146384],
    },
    { fn: "mape", mo: [0.5, 0.3, 0.2], w: false, val: 56.82293848309868 },
    { fn: "mape", mo: [0.5, 0.3, 0.2], w: true, val: 20.73711852255354 },
    { fn: "medae", mo: "uniform_average", w: false, val: 0.5576666666666666 },
    {
      fn: "medae",
      mo: "raw_values",
      w: false,
      val: [0.5350000000000001, 1.0719999999999998, 0.06600000000000002],
    },
    { fn: "medae", mo: [0.5, 0.3, 0.2], w: false, val: 0.6023 },
    {
      fn: "r2const",
      mo: "raw_values",
      w: false,
      val: [0.837719779338407, 0.0, 0.6049325128707708],
    },
    { fn: "r2const", mo: "variance_weighted", w: false, val: 0.8363812134922722 },
    { fn: "r2const", mo: "uniform_average", w: false, val: 0.48088409740305926 },
  ],
  yt2: [
    [5.072, 4.0, 0.01],
    [1.223, 4.0, 0.001],
    [-0.003, 4.0, 0.305],
    [1.801, 4.0, -0.052],
    [1.516, 4.0, -0.073],
    [-4.36, 4.0, 0.037],
    [0.823, 4.0, 0.495],
  ],
  yp2: [
    [5.226, -1.785, 0.213],
    [1.178, -3.818, -0.04],
    [-2.291, -4.215, 0.264],
    [1.058, -0.804, -0.217],
    [2.051, -2.848, -0.139],
    [-5.564, 3.126, 0.214],
    [0.494, -3.739, 0.477],
  ],
  raw1d: [1.0938308571428568],
};

type Options = {
  sampleWeight?: number[];
  multioutput?: "uniformAverage" | "rawValues" | "varianceWeighted" | number[];
};
type Metric = (
  yTrue: ReturnType<typeof tensor>,
  yPred: ReturnType<typeof tensor>,
  o?: Options
) => number | number[];

const METRICS: Record<string, Metric> = {
  mse: (a, b, o) => mse(a, b, o as never),
  rmse: (a, b, o) => rmse(a, b, o as never),
  mae: (a, b, o) => mae(a, b, o as never),
  r2: (a, b, o) => r2Score(a, b, o as never),
  ev: (a, b, o) => explainedVarianceScore(a, b, o as never),
  mape: (a, b, o) => meanAbsolutePercentageError(a, b, o as never),
  medae: (a, b, o) => medianAbsoluteError(a, b, o as never),
};

const MODE: Record<string, "uniformAverage" | "rawValues" | "varianceWeighted"> = {
  uniform_average: "uniformAverage",
  raw_values: "rawValues",
  variance_weighted: "varianceWeighted",
};

function expectClose(actual: number | number[], expected: number | number[]) {
  if (Array.isArray(expected)) {
    expect(Array.isArray(actual)).toBe(true);
    expect((actual as number[]).length).toBe(expected.length);
    for (const [i, e] of expected.entries()) expect((actual as number[])[i]).toBeCloseTo(e, 10);
  } else {
    expect(typeof actual).toBe("number");
    expect(actual as number).toBeCloseTo(expected, 10);
  }
}

describe("regression metrics with several outputs match scikit-learn", () => {
  const yTrue = tensor(REF.yt);
  const yPred = tensor(REF.yp);
  for (const r of REF.res) {
    if (r.fn === "r2const") continue;
    const label = `${r.fn} ${JSON.stringify(r.mo)}${r.w ? " weighted" : ""}`;
    it(label, () => {
      const options: Options = {
        multioutput: Array.isArray(r.mo) ? r.mo : (MODE[r.mo] as "uniformAverage"),
        ...(r.w ? { sampleWeight: REF.sw } : {}),
      };
      expectClose((METRICS[r.fn] as Metric)(yTrue, yPred, options), r.val);
    });
  }

  it("scores a constant output as scikit-learn does", () => {
    const yt2 = tensor(REF.yt2);
    const yp2 = tensor(REF.yp2);
    for (const r of REF.res.filter((x) => x.fn === "r2const")) {
      expectClose(r2Score(yt2, yp2, { multioutput: MODE[r.mo as string] as never }), r.val);
    }
  });

  it("uses the mean over outputs by default", () => {
    const expected = REF.res.find((r) => r.fn === "mse" && r.mo === "uniform_average" && !r.w);
    expectClose(mse(yTrue, yPred), expected?.val as number);
  });

  it("returns one value for a 1-D target with rawValues", () => {
    const t = tensor(REF.yt.map((row) => row[0] as number));
    const p = tensor(REF.yp.map((row) => row[0] as number));
    expectClose(mse(t, p, { multioutput: "rawValues" }), REF.raw1d);
    expectClose(mse(t, p), REF.raw1d[0] as number);
  });

  it("keeps column vectors working as before", () => {
    const t = tensor(REF.yt.map((row) => [row[0] as number]));
    const p = tensor(REF.yp.map((row) => [row[0] as number]));
    expectClose(mse(t, p), REF.raw1d[0] as number);
  });

  it("rejects bad multioutput values and shapes", () => {
    expect(() => mse(yTrue, yPred, { multioutput: [1, 2] })).toThrow(InvalidParameterError);
    expect(() => mse(yTrue, yPred, { multioutput: [0, 0, 0] })).toThrow(InvalidParameterError);
    expect(() => mse(yTrue, yPred, { multioutput: "nope" as never })).toThrow(
      InvalidParameterError
    );
    expect(() => mse(yTrue, yPred, { multioutput: "varianceWeighted" as never })).toThrow(
      InvalidParameterError
    );
    expect(() => mse(yTrue, tensor(REF.yp.map((row) => row.slice(0, 2))))).toThrow(ShapeError);
    expect(() => mse(tensor([1, 2, 3]), tensor(REF.yp.slice(0, 3)))).toThrow(ShapeError);
    expect(() => mse(tensor([["a"]]), tensor([["a"]]))).toThrow(DTypeError);
  });
});
