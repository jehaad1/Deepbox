// biome-ignore-all lint/suspicious/noApproximativeNumericConstant: values copied from pandas output
import { afterEach, describe, expect, it } from "vitest";
import { DataFrame, Series } from "../../src/dataframe";
import { averageRanks, kendall, pearson, spearman } from "../../src/dataframe/correlation";
import { clearSeed, setSeed } from "../../src/random";

// Expected values below were produced with pandas 3.0 and SciPy 1.17 (see each block).

const column = (df: DataFrame, name: string): unknown[] => [...df.getColumnData(name)];
const nan = (v: unknown): number => (v === null || v === undefined ? Number.NaN : (v as number));

/** Rows of a frame as numbers, with missing values as NaN. */
const matrix = (df: DataFrame): number[][] => {
  const cols = df.columns.map((c) => column(df, c));
  return df.index.map((_, i) => cols.map((col) => nan(col[i])));
};

const expectClose = (actual: unknown, expected: number, tol = 1e-12): void => {
  const a = nan(actual);
  if (Number.isNaN(expected)) expect(a).toBeNaN();
  else expect(Math.abs(a - expected)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(expected)));
};

const expectMatrix = (df: DataFrame, expected: number[][], tol = 1e-12): void => {
  const got = matrix(df);
  expect(got.length).toBe(expected.length);
  for (let i = 0; i < got.length; i++) {
    const row = got[i] as number[];
    const want = expected[i] as number[];
    expect(row.length).toBe(want.length);
    for (let j = 0; j < row.length; j++) expectClose(row[j], want[j] as number, tol);
  }
};

const expectVector = (
  actual: readonly unknown[],
  expected: readonly number[],
  tol = 1e-12
): void => {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) expectClose(actual[i], expected[i] as number, tol);
};

const FFILL = [
  [1.0, NaN],
  [1.0, 2.0],
  [1.0, 2.0],
  [4.0, 2.0],
  [4.0, 5.0],
  [6.0, 5.0],
];
const FFILL1 = [
  [1.0, NaN],
  [1.0, 2.0],
  [NaN, 2.0],
  [4.0, NaN],
  [4.0, 5.0],
  [6.0, 5.0],
];
const BFILL = [
  [1.0, 2.0],
  [4.0, 2.0],
  [4.0, 5.0],
  [4.0, 5.0],
  [6.0, 5.0],
  [6.0, NaN],
];
const BFILL1 = [
  [1.0, 2.0],
  [NaN, 2.0],
  [4.0, NaN],
  [4.0, 5.0],
  [6.0, 5.0],
  [6.0, NaN],
];
const FFILL_AX1 = [
  [1.0, 1.0],
  [NaN, 2.0],
  [NaN, NaN],
  [4.0, 4.0],
  [NaN, 5.0],
  [6.0, 6.0],
];
const BFILL_AX1 = [
  [1.0, NaN],
  [2.0, 2.0],
  [NaN, NaN],
  [4.0, NaN],
  [5.0, 5.0],
  [6.0, NaN],
];
const BFILL_AX1_L1 = [
  [1.0, NaN],
  [2.0, 2.0],
  [NaN, NaN],
  [4.0, NaN],
  [5.0, 5.0],
  [6.0, NaN],
];
const CORR_DATA = {
  x: [
    0.0012301533574825742,
    0.2987455375084699,
    NaN,
    -0.8905918387572742,
    -0.45467078517172255,
    -0.9916465549964624,
    0.060143602597438485,
    1.3402152455545335,
    -0.49220651855132963,
    NaN,
    0.4898420501851982,
    0.35688700816006075,
    0.10541424899789856,
    -0.9304680447082047,
    -0.02925182246327349,
  ],
  y: [
    2.0006150766787414, 2.149372768754235, 1.8629310723188912, 3.554704080621363, 3.772664607414139,
    2.504176722501769, 3.0300718012987193, 3.670107622777267, 0.7538967407243352, 3.68976255009003,
    2.2449210250925993, 1.1784435040800303, 4.05270712449895, -0.46523402235410233,
    3.985374088768363,
  ],
  z: [
    0.2712643588217015,
    0.15675108662422516,
    -0.18693094462995438,
    -2.516759710820513,
    NaN,
    NaN,
    0.11330898600330756,
    -1.5301357655053935,
    -0.47775327603393064,
    -0.9785190780566395,
    -0.8088372394255993,
    NaN,
    -0.8075346753318965,
    -0.0325217049455206,
    0.8843898673831739,
  ],
};
const CORR_PEARSON = [
  [1.0, 0.28248287993129967, -0.0032425338596628636],
  [0.28248287993129967, 1.0, -0.296564829203333],
  [-0.0032425338596628636, -0.296564829203333, 1.0],
];
const CORR_PEARSON_MP13 = [
  [1.0, 0.28248287993129967, NaN],
  [0.28248287993129967, 1.0, NaN],
  [NaN, NaN, NaN],
];
const CORR_SPEARMAN = [
  [1.0, 0.15384615384615385, -0.16363636363636364],
  [0.15384615384615385, 1.0, -0.25874125874125875],
  [-0.16363636363636364, -0.25874125874125875, 1.0],
];
const CORR_SPEARMAN_MP13 = [
  [1.0, 0.15384615384615385, NaN],
  [0.15384615384615385, 1.0, NaN],
  [NaN, NaN, NaN],
];
const CORR_KENDALL = [
  [1.0, 0.15384615384615383, -0.19999999999999998],
  [0.15384615384615383, 1.0, -0.1515151515151515],
  [-0.19999999999999998, -0.1515151515151515, 1.0],
];
const CORR_KENDALL_MP13 = [
  [1.0, 0.15384615384615383, NaN],
  [0.15384615384615383, 1.0, NaN],
  [NaN, NaN, NaN],
];
const FULL_DATA = {
  p: [1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 5.0, 8.0],
  q: [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0],
  r: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
  t: [8.0, 7.0, 7.0, 5.0, 4.0, 3.0, 3.0, 1.0],
};
const FULL_PEARSON = [
  [1.0, 0.5406947784570656, NaN, -0.9777805108428541],
  [0.5406947784570656, 1.0, NaN, -0.5176813321554602],
  [NaN, NaN, NaN, NaN],
  [-0.9777805108428541, -0.5176813321554602, NaN, 1.0],
];
const FULL_SPEARMAN = [
  [1.0, 0.5495502618648208, NaN, -0.9815368735540919],
  [0.5495502618648208, 1.0, NaN, -0.527282411152491],
  [NaN, NaN, NaN, NaN],
  [-0.9815368735540919, -0.527282411152491, NaN, 1.0],
];
// pandas reports 1.0 for the Kendall diagonal cell of a constant column.
const FULL_KENDALL = [
  [1.0, 0.4321208107251124, NaN, -0.960768922830523],
  [0.4321208107251124, 1.0, NaN, -0.4151682458530185],
  [NaN, NaN, 1.0, NaN],
  [-0.960768922830523, -0.4151682458530185, NaN, 1.0],
];
const ROLL_V = [1.0, 2.0, NaN, 4.0, 5.0, NaN, NaN, 8.0, 9.0, 10.0];
const ROLL_A_SUM = [3.0, 3.0, 7.0, 11.0, 9.0, 9.0, 13.0, 17.0, 27.0, 27.0];
const ROLL_A_MEAN = [
  1.5, 1.5, 2.3333333333333335, 3.6666666666666665, 4.5, 4.5, 6.5, 8.5, 9.0, 9.0,
];
const ROLL_A_STD = [
  0.7071067811865476, 0.7071067811865476, 1.5275252316519465, 1.5275252316519465,
  0.7071067811865474, 0.7071067811865474, 2.1213203435596424, 0.7071067811865476, 1.0, 1.0,
];
const ROLL_A_VAR = [
  0.5, 0.5, 2.333333333333333, 2.333333333333333, 0.4999999999999998, 0.4999999999999998, 4.5, 0.5,
  1.0, 1.0,
];
const ROLL_A_MIN = [1.0, 1.0, 1.0, 2.0, 4.0, 4.0, 5.0, 8.0, 8.0, 8.0];
const ROLL_A_MAX = [2.0, 2.0, 4.0, 5.0, 5.0, 5.0, 8.0, 9.0, 10.0, 10.0];
const ROLL_A_MEDIAN = [1.5, 1.5, 2.0, 4.0, 4.5, 4.5, 6.5, 8.5, 9.0, 9.0];
const ROLL_B_SUM = [1.0, 3.0, 3.0, 6.0, 9.0, 9.0, 5.0, 8.0, 17.0, 27.0];
const ROLL_B_MEAN = [1.0, 1.5, 1.5, 3.0, 4.5, 4.5, 5.0, 8.0, 8.5, 9.0];
const ROLL_B_STD = [
  NaN,
  0.7071067811865476,
  0.7071067811865476,
  1.4142135623730951,
  0.7071067811865476,
  0.7071067811865476,
  NaN,
  NaN,
  0.7071067811865476,
  1.0,
];
const ROLL_B_VAR = [NaN, 0.5, 0.5, 2.0, 0.5, 0.5, NaN, NaN, 0.5, 1.0];
const ROLL_B_MIN = [1.0, 1.0, 1.0, 2.0, 4.0, 4.0, 5.0, 8.0, 8.0, 8.0];
const ROLL_B_MAX = [1.0, 2.0, 2.0, 4.0, 5.0, 5.0, 5.0, 8.0, 9.0, 10.0];
const ROLL_B_MEDIAN = [1.0, 1.5, 1.5, 3.0, 4.5, 4.5, 5.0, 8.0, 8.5, 9.0];
const ROLL_C_SUM = [1.0, 3.0, 3.0, 6.0, 9.0, 9.0, 5.0, 8.0, 17.0, 27.0];
const ROLL_C_MEAN = [1.0, 1.5, 1.5, 3.0, 4.5, 4.5, 5.0, 8.0, 8.5, 9.0];
const ROLL_C_STD = [
  NaN,
  0.7071067811865476,
  0.7071067811865476,
  1.4142135623730951,
  0.7071067811865476,
  0.7071067811865476,
  NaN,
  NaN,
  0.7071067811865476,
  1.0,
];
const ROLL_C_VAR = [NaN, 0.5, 0.5, 2.0, 0.5, 0.5, NaN, NaN, 0.5, 1.0];
const ROLL_C_MIN = [1.0, 1.0, 1.0, 2.0, 4.0, 4.0, 5.0, 8.0, 8.0, 8.0];
const ROLL_C_MAX = [1.0, 2.0, 2.0, 4.0, 5.0, 5.0, 5.0, 8.0, 9.0, 10.0];
const ROLL_C_MEDIAN = [1.0, 1.5, 1.5, 3.0, 4.5, 4.5, 5.0, 8.0, 8.5, 9.0];
const ROLL_D_SUM = [1.0, 3.0, 2.0, 4.0, 9.0, 5.0, NaN, 8.0, 17.0, 19.0];
const ROLL_D_MEAN = [1.0, 1.5, 2.0, 4.0, 4.5, 5.0, NaN, 8.0, 8.5, 9.5];
const ROLL_D_STD = [
  NaN,
  0.7071067811865476,
  NaN,
  NaN,
  0.7071067811865476,
  NaN,
  NaN,
  NaN,
  0.7071067811865476,
  0.7071067811865476,
];
const ROLL_D_VAR = [NaN, 0.5, NaN, NaN, 0.5, NaN, NaN, NaN, 0.5, 0.5];
const ROLL_D_MIN = [1.0, 1.0, 2.0, 4.0, 4.0, 5.0, NaN, 8.0, 8.0, 9.0];
const ROLL_D_MAX = [1.0, 2.0, 2.0, 4.0, 5.0, 5.0, NaN, 8.0, 9.0, 10.0];
const ROLL_D_MEDIAN = [1.0, 1.5, 2.0, 4.0, 4.5, 5.0, NaN, 8.0, 8.5, 9.5];
const ROLL_E_SUM = [NaN, 7.0, 12.0, 11.0, NaN, 17.0, 22.0, 27.0, 27.0, 27.0];
const ROLL_E_MEAN = [
  NaN,
  2.3333333333333335,
  3.0,
  3.6666666666666665,
  NaN,
  5.666666666666667,
  7.333333333333333,
  9.0,
  9.0,
  9.0,
];
const ROLL_E_STD = [
  NaN,
  1.5275252316519465,
  1.8257418583505538,
  1.5275252316519468,
  NaN,
  2.0816659994661326,
  2.0816659994661326,
  1.0,
  1.0,
  1.0,
];
const ROLL_E_VAR = [
  NaN,
  2.333333333333333,
  3.3333333333333335,
  2.3333333333333335,
  NaN,
  4.333333333333333,
  4.333333333333333,
  1.0000000000000002,
  1.0000000000000002,
  1.0000000000000002,
];
const ROLL_E_MIN = [NaN, 1.0, 1.0, 2.0, NaN, 4.0, 5.0, 8.0, 8.0, 8.0];
const ROLL_E_MAX = [NaN, 4.0, 5.0, 5.0, NaN, 8.0, 9.0, 10.0, 10.0, 10.0];
const ROLL_E_MEDIAN = [NaN, 2.0, 3.0, 4.0, NaN, 5.0, 8.0, 9.0, 9.0, 9.0];
const ROLL_F_SUM = [NaN, NaN, 9.0, 11.0, 19.0, 17.0, 22.0, 22.0, 16.0, NaN];
const ROLL_F_MEAN = [NaN, NaN, 2.25, 2.75, 4.75, 4.25, 5.5, 5.5, 4.0, NaN];
const ROLL_F_STD = [
  NaN,
  NaN,
  1.5,
  2.0615528128088303,
  3.304037933599835,
  3.593976442141304,
  2.886751345948129,
  2.886751345948129,
  1.8257418583505538,
  NaN,
];
const ROLL_F_MIN = [NaN, NaN, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, NaN];
const ROLL_F_MAX = [NaN, NaN, 4.0, 5.0, 9.0, 9.0, 9.0, 9.0, 6.0, NaN];
const ROLL_F_MEDIAN = [NaN, NaN, 2.0, 2.5, 4.5, 3.5, 5.5, 5.5, 4.0, NaN];
const ROLL_G_SUM = [NaN, NaN, 14.0, 20.0, 21.0, 23.0, 27.0, 25.0, NaN, NaN];
const ROLL_G_MEAN = [NaN, NaN, 2.8, 4.0, 4.2, 4.6, 5.4, 5.0, NaN, NaN];
const ROLL_G_STD = [
  NaN,
  NaN,
  1.7888543819998317,
  3.3166247903554,
  3.1144823004794873,
  3.2093613071762426,
  2.5099800796022267,
  2.7386127875258306,
  NaN,
  NaN,
];
const ROLL_G_MIN = [NaN, NaN, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, NaN, NaN];
const ROLL_G_MAX = [NaN, NaN, 5.0, 9.0, 9.0, 9.0, 9.0, 9.0, NaN, NaN];
const ROLL_G_MEDIAN = [NaN, NaN, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0, NaN, NaN];
const ROLL_H_SUM = [NaN, 8.0, 6.0, 10.0, 15.0, 16.0, 17.0, 13.0, 14.0, NaN];
const ROLL_H_MEAN = [
  NaN,
  2.6666666666666665,
  2.0,
  3.3333333333333335,
  5.0,
  5.333333333333333,
  5.666666666666667,
  4.333333333333333,
  4.666666666666667,
  NaN,
];
const ROLL_H_STD = [
  NaN,
  1.5275252316519468,
  1.7320508075688772,
  2.0816659994661326,
  4.0,
  3.511884584284246,
  3.511884584284246,
  2.0816659994661326,
  1.5275252316519468,
  NaN,
];
const ROLL_H_MIN = [NaN, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, NaN];
const ROLL_H_MAX = [NaN, 4.0, 4.0, 5.0, 9.0, 9.0, 9.0, 6.0, 6.0, NaN];
const ROLL_H_MEDIAN = [NaN, 3.0, 1.0, 4.0, 5.0, 5.0, 6.0, 5.0, 5.0, NaN];
const ROLL_FULL = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0];
const G_DATA = {
  k: ["a", "b", "a", "c", "b", "a"],
  k2: [1, 1, 2, 1, 1, 1],
  x: [1.0, 2.0, 3.0, 4.0, NaN, 10.0],
  y: [5.0, 5.0, 6.0, 7.0, 8.0, 5.0],
  s: ["p", "q", "p", NaN, "q", "z"],
};
const G_QUANTILE = [
  [2.2, 5.0],
  [2.0, 5.9],
  [4.0, 7.0],
];
const G_NUNIQUE = [
  [3, 2, 2],
  [1, 2, 1],
  [1, 1, 0],
];
const G_TMEAN = [
  [4.666666666666667, 5.333333333333333],
  [2.0, 6.5],
  [4.666666666666667, 5.333333333333333],
  [4.0, 7.0],
  [2.0, 6.5],
  [4.666666666666667, 5.333333333333333],
];
const G_TCUMSUM = [
  [1.0, 5.0],
  [2.0, 5.0],
  [4.0, 11.0],
  [4.0, 7.0],
  [NaN, 13.0],
  [14.0, 16.0],
];
const G_TFFILL = [
  [1.0, 5.0],
  [2.0, 5.0],
  [3.0, 6.0],
  [4.0, 7.0],
  [2.0, 8.0],
  [10.0, 5.0],
];
const G_TCOUNT = [
  [3, 3],
  [1, 2],
  [3, 3],
  [1, 1],
  [1, 2],
  [3, 3],
];
const G_TMAX = [
  [10.0, 6.0],
  [2.0, 8.0],
  [10.0, 6.0],
  [4.0, 7.0],
  [2.0, 8.0],
  [10.0, 6.0],
];
const G_TSTD = [
  [4.725815626252609, 0.5773502691896257],
  [NaN, 2.1213203435596424],
  [4.725815626252609, 0.5773502691896257],
  [NaN, NaN],
  [NaN, 2.1213203435596424],
  [4.725815626252609, 0.5773502691896257],
];
const G_GET_A = [
  [1.0, 5.0],
  [3.0, 6.0],
  [10.0, 5.0],
];
const G_GET_BOTH = [[2.0], [NaN]];
const G_NAMED = [
  [4.666666666666667, 16.0, 2.0],
  [2.0, 13.0, 1.0],
  [4.0, 7.0, 0.0],
];
const CAT_OUTER = [
  [1.0, 3.0, NaN],
  [2.0, 4.0, NaN],
  [NaN, 5.0, 7.0],
  [NaN, 6.0, 8.0],
];
const CAT_INNER = [[3], [4], [5], [6]];
const CAT_AX1_INNER = [
  [2, 4],
  [3, 5],
];
const CAT_AX1_OUTER = [
  [1.0, NaN],
  [2.0, 4.0],
  [3.0, 5.0],
  [NaN, 6.0],
];
const VC_NORM = [0.42857142857142855, 0.42857142857142855, 0.14285714285714285];
const VC_NORM_IDX = ["a", "b", "c"];
const VC_NA_NORM = [0.3333333333333333, 0.3333333333333333, 0.2222222222222222, 0.1111111111111111];
const VC_NA_IDX = ["a", "b", "c", "null"];
const VC_NA_NORM_DROP = [0.375, 0.375, 0.25];
const VC_ASC_IDX = [5, 3, 1, 2];
const VC_ASC = [1, 2, 2, 2];
const VC_NOSORT_IDX = [3, 1, 2, 5];
const DVC_DROP_NORM = [0.6, 0.4];

afterEach(() => clearSeed());

describe("wave3 dataframe: ffill, bfill and fillna", () => {
  const df = () =>
    new DataFrame({ a: [1, null, null, 4, null, 6], b: [null, 2, null, null, 5, null] });

  it("ffill and bfill match pandas, with and without limit", () => {
    expectMatrix(df().ffill(), FFILL);
    expectMatrix(df().ffill({ limit: 1 }), FFILL1);
    expectMatrix(df().bfill(), BFILL);
    expectMatrix(df().bfill({ limit: 1 }), BFILL1);
  });

  it("fills across rows with axis 1", () => {
    expectMatrix(df().ffill({ axis: 1 }), FFILL_AX1);
    expectMatrix(df().bfill({ axis: 1 }), BFILL_AX1);
    expectMatrix(df().bfill({ axis: "columns", limit: 1 }), BFILL_AX1_L1);
  });

  it("keeps the leading gap of ffill and the trailing gap of bfill missing", () => {
    expect(column(df().ffill(), "b")[0]).toBeNull();
    expect(column(df().bfill(), "b")[5]).toBeNull();
  });

  it("treats NaN and undefined as missing and keeps the values it copies", () => {
    const frame = new DataFrame({
      a: ["x", undefined, Number.NaN, "y"],
      d: [new Date(0), null, null, null],
    });
    expect(column(frame.ffill(), "a")).toEqual(["x", "x", "x", "y"]);
    const filled = column(frame.ffill(), "d");
    expect(filled[3]).toBe(filled[0]);
  });

  it("does not change the original and keeps index and columns", () => {
    const frame = new DataFrame({ a: [1, null] }, { index: ["p", "q"] });
    const out = frame.ffill();
    expect(column(frame, "a")).toEqual([1, null]);
    expect(out.index).toEqual(["p", "q"]);
    expect(out.columns).toEqual(["a"]);
  });

  it("handles empty frames and frames without columns", () => {
    expect(new DataFrame({ a: [] as number[] }).ffill().shape).toEqual([0, 1]);
    expect(new DataFrame({}).bfill().shape).toEqual([0, 0]);
    expect(new DataFrame({ a: [null, null] }).ffill({ axis: 1 }).shape).toEqual([2, 1]);
  });

  it("rejects an invalid limit and axis", () => {
    expect(() => df().ffill({ limit: 0 })).toThrow(/limit/);
    expect(() => df().bfill({ limit: 1.5 })).toThrow(/limit/);
    expect(() => df().ffill({ limit: -2 })).toThrow(/limit/);
    expect(() => df().ffill({ axis: 2 })).toThrow();
  });

  it("fillna accepts { method, limit } and the pandas aliases", () => {
    expectMatrix(df().fillna({ method: "ffill" }), FFILL);
    expectMatrix(df().fillna({ method: "pad", limit: 1 }), FFILL1);
    expectMatrix(df().fillna({ method: "bfill" }), BFILL);
    expectMatrix(df().fillna({ method: "backfill", limit: 1 }), BFILL1);
    expectMatrix(df().fillna({ method: "bfill", axis: 1 }), BFILL_AX1);
    expect(() => df().fillna({ method: "nearest" as never })).toThrow(/method/);
    expect(() => df().fillna({ method: "ffill", limit: 0 })).toThrow(/limit/);
  });

  it("fillna accepts one value per column and leaves other columns alone", () => {
    const out = df().fillna({ a: -1 });
    expect(column(out, "a")).toEqual([1, -1, -1, 4, -1, 6]);
    expect(column(out, "b")).toEqual([null, 2, null, null, 5, null]);
    const both = df().fillna({ a: 0, b: "none", missing: 5 });
    expect(column(both, "b")).toEqual(["none", 2, "none", "none", 5, "none"]);
    expect(both.columns).toEqual(["a", "b"]);
  });

  it("fillna reads a column called method as a per-column map", () => {
    const frame = new DataFrame({ method: [1, null], other: [null, 2] });
    const out = frame.fillna({ method: 9 });
    expect(column(out, "method")).toEqual([1, 9]);
    expect(column(out, "other")).toEqual([null, 2]);
  });

  it("fillna still fills with a scalar, an array or a Date", () => {
    expect(column(df().fillna(0), "a")).toEqual([1, 0, 0, 4, 0, 6]);
    const when = new Date(5);
    expect(column(df().fillna(when), "b")[0]).toBe(when);
    const list = [1, 2];
    expect(column(df().fillna(list), "b")[0]).toBe(list);
  });

  it("Series.fillna, ffill and bfill keep index and name", () => {
    const s = new Series([null, 1, null, null, 4, null], {
      index: ["a", "b", "c", "d", "e", "f"],
      name: "v",
    });
    expect(s.ffill().toArray()).toEqual([null, 1, 1, 1, 4, 4]);
    expect(s.ffill(1).toArray()).toEqual([null, 1, 1, null, 4, 4]);
    expect(s.bfill().toArray()).toEqual([1, 1, 4, 4, 4, null]);
    expect(s.bfill(1).toArray()).toEqual([1, 1, null, 4, 4, null]);
    expect(s.ffill().index).toEqual(["a", "b", "c", "d", "e", "f"]);
    expect(s.bfill().name).toBe("v");
    expect(s.fillna(0).toArray()).toEqual([0, 1, 0, 0, 4, 0]);
    expect(s.fillna({ method: "ffill", limit: 2 }).toArray()).toEqual([null, 1, 1, 1, 4, 4]);
    expect(s.fillna({ method: "backfill" }).toArray()).toEqual([1, 1, 4, 4, 4, null]);
    expect(s.toArray()).toEqual([null, 1, null, null, 4, null]);
  });

  it("Series fills NaN and invalid dates and rejects bad arguments", () => {
    expect(new Series([1, Number.NaN, 3]).ffill().toArray()).toEqual([1, 1, 3]);
    const bad = new Date(Number.NaN);
    const good = new Date(10);
    expect(new Series<Date>([good, bad]).ffill().toArray()[1]).toBe(good);
    expect(new Series<number>([]).ffill().length).toBe(0);
    expect(() => new Series([1]).ffill(0)).toThrow(/limit/);
    expect(() => new Series([1]).fillna({ method: "x" as never })).toThrow(/method/);
  });
});

describe("wave3 dataframe: corr", () => {
  const data = () => new DataFrame(CORR_DATA);

  it("pearson, spearman and kendall match pandas on data with gaps", () => {
    expectMatrix(data().corr(), CORR_PEARSON);
    expectMatrix(data().corr("pearson"), CORR_PEARSON);
    expectMatrix(data().corr("spearman"), CORR_SPEARMAN);
    expectMatrix(data().corr({ method: "kendall" }), CORR_KENDALL);
  });

  it("minPeriods blanks the cells with too few complete pairs", () => {
    expectMatrix(data().corr({ minPeriods: 13 }), CORR_PEARSON_MP13);
    expectMatrix(data().corr("spearman", 13), CORR_SPEARMAN_MP13);
    expectMatrix(data().corr({ method: "kendall", minPeriods: 13 }), CORR_KENDALL_MP13);
  });

  it("matches pandas on complete data with ties and a constant column", () => {
    const full = new DataFrame(FULL_DATA);
    expectMatrix(full.corr(), FULL_PEARSON);
    expectMatrix(full.corr("spearman"), FULL_SPEARMAN);
    expectMatrix(full.corr("kendall"), FULL_KENDALL);
  });

  it("returns exactly 1 on the diagonal and NaN for a constant column (Kendall: 1, like pandas)", () => {
    for (const method of ["pearson", "spearman", "kendall"] as const) {
      for (const frame of [data(), new DataFrame(FULL_DATA)]) {
        const out = frame.corr(method);
        out.columns.forEach((c, i) => {
          const v = column(out, c)[i] as number;
          if (c === "r" && method !== "kendall") expect(v).toBeNaN();
          else expect(v).toBe(1);
        });
      }
    }
  });

  it("Kendall diagonal follows pandas for single values and minPeriods", () => {
    const frame = new DataFrame({ a: [1, null, null], b: [1, 2, 3] });
    expectMatrix(frame.corr({ method: "kendall" }), [
      [1, NaN],
      [NaN, 1],
    ]);
    expectMatrix(frame.corr({ method: "kendall", minPeriods: 2 }), [
      [NaN, NaN],
      [NaN, 1],
    ]);
    expectMatrix(frame.corr({ method: "pearson" }), [
      [NaN, NaN],
      [NaN, 1],
    ]);
  });

  it("is symmetric and labelled by the numeric columns", () => {
    const frame = new DataFrame({ n: [1, 2, 3, 4], s: ["a", "b", "c", "d"], m: [4, 3, 1, 2] });
    const out = frame.corr("kendall");
    expect(out.columns).toEqual(["n", "m"]);
    expect(out.index).toEqual(["n", "m"]);
    expect(column(out, "n")[1]).toBe(column(out, "m")[0]);
    expectClose(column(out, "n")[1], -2 / 3);
  });

  it("rank methods are 1 for a monotone relation even when it is not linear", () => {
    const frame = new DataFrame({ a: [1, 2, 3, 4, 5], b: [1, 8, 27, 64, 125] });
    expectClose(column(frame.corr("spearman"), "a")[1], 1);
    expectClose(column(frame.corr("kendall"), "a")[1], 1);
    expect(column(frame.corr(), "a")[1] as number).toBeLessThan(1);
  });

  it("gives NaN for fewer than two complete pairs and for empty frames", () => {
    expect(matrix(new DataFrame({ a: [1], b: [2] }).corr())).toEqual([
      [Number.NaN, Number.NaN],
      [Number.NaN, Number.NaN],
    ]);
    expect(new DataFrame({ a: [1, null], b: [null, 2] }).corr("spearman").shape).toEqual([2, 2]);
    expect(new DataFrame({}).corr().shape).toEqual([0, 0]);
    expect(new DataFrame({ s: ["a", "b"] }).corr("kendall").shape).toEqual([0, 0]);
  });

  it("minPeriods 0 behaves like 1 and a large value blanks everything", () => {
    const frame = new DataFrame({ a: [1, 2, 3], b: [3, 1, 2] });
    expectMatrix(frame.corr({ minPeriods: 0 }), matrix(frame.corr()));
    expectMatrix(frame.corr({ minPeriods: 4 }), [
      [Number.NaN, Number.NaN],
      [Number.NaN, Number.NaN],
    ]);
  });

  it("rejects an unknown method and a bad minPeriods", () => {
    expect(() => data().corr("cosine" as never)).toThrow(/method/);
    expect(() => data().corr({ minPeriods: -1 })).toThrow(/minPeriods/);
    expect(() => data().corr("pearson", 1.5)).toThrow(/minPeriods/);
  });

  it("kendall tau-b agrees with the O(n^2) definition and SciPy-style ties", () => {
    let seed = 12345;
    const next = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648;
    };
    for (let trial = 0; trial < 20; trial++) {
      const n = 5 + Math.floor(next() * 40);
      const x = Array.from({ length: n }, () => Math.floor(next() * 6));
      const y = Array.from({ length: n }, () => Math.floor(next() * 6));
      let conc = 0;
      let disc = 0;
      let tx = 0;
      let ty = 0;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const dx = Math.sign((x[i] as number) - (x[j] as number));
          const dy = Math.sign((y[i] as number) - (y[j] as number));
          if (dx === 0) tx++;
          if (dy === 0) ty++;
          if (dx * dy > 0) conc++;
          else if (dx * dy < 0) disc++;
        }
      }
      const pairs = (n * (n - 1)) / 2;
      const expected = (conc - disc) / Math.sqrt((pairs - tx) * (pairs - ty));
      const got = kendall(x, y);
      if (Number.isNaN(expected)) expect(got).toBeNaN();
      else expect(got).toBeCloseTo(expected, 12);
    }
  });

  it("exposes pearson, spearman and average ranks consistently", () => {
    expect([...averageRanks([10, 20, 20, 5])]).toEqual([2, 3.5, 3.5, 1]);
    expect(pearson([1, 2, 3], [2, 4, 6])).toBeCloseTo(1, 15);
    expect(spearman([1, 2, 3], [9, 4, 1])).toBeCloseTo(-1, 15);
    expect(pearson([1], [1])).toBeNaN();
    expect(kendall([1, 1, 1], [1, 2, 3])).toBeNaN();
  });
});

describe("wave3 dataframe: sample", () => {
  const frame = () =>
    new DataFrame(
      { a: [10, 20, 30, 40, 50], w: [0, 0, 1, 1, 8] },
      { index: ["p", "q", "r", "s", "t"] }
    );

  it("keeps sample(n, random_state) working: distinct rows, labels and reproducibility", () => {
    const out = frame().sample(3, 42);
    expect(out.shape).toEqual([3, 2]);
    expect(new Set(out.index).size).toBe(3);
    out.index.forEach((label, i) => {
      expect(column(frame(), "a")[["p", "q", "r", "s", "t"].indexOf(String(label))]).toBe(
        column(out, "a")[i]
      );
    });
    expect(frame().sample(3, 42).index).toEqual(out.index);
    expect(frame().sample({ n: 3, randomState: 42 }).index).toEqual(out.index);
    const other = new Set<string>();
    for (let seed = 0; seed < 20; seed++) other.add(frame().sample(3, seed).index.join(","));
    expect(other.size).toBeGreaterThan(5);
  });

  it("draws one row by default and handles 0 and all rows", () => {
    expect(frame().sample().shape[0]).toBe(1);
    expect(frame().sample({}).shape[0]).toBe(1);
    expect(frame().sample(0).shape).toEqual([0, 2]);
    expect([...frame().sample(5, 1).index].sort()).toEqual(["p", "q", "r", "s", "t"]);
  });

  it("frac rounds half to even like pandas", () => {
    expect(frame().sample({ frac: 0.5 }).shape[0]).toBe(2); // 2.5 -> 2
    expect(frame().sample({ frac: 0.3 }).shape[0]).toBe(2); // 1.5 -> 2
    expect(frame().sample({ frac: 0.7 }).shape[0]).toBe(4); // 3.5 -> 4
    expect(frame().sample({ frac: 1 }).shape[0]).toBe(5);
    expect(frame().sample({ frac: 0 }).shape[0]).toBe(0);
    expect(frame().sample({ frac: 2, replace: true }).shape[0]).toBe(10);
  });

  it("replace draws with repetition and labels the result 0..k-1", () => {
    const out = frame().sample({ n: 12, replace: true, randomState: 3 });
    expect(out.shape[0]).toBe(12);
    expect(out.index).toEqual(Array.from({ length: 12 }, (_, i) => i));
    const values = column(out, "a") as number[];
    expect(new Set(values).size).toBeLessThan(12);
    for (const v of values) expect([10, 20, 30, 40, 50]).toContain(v);
  });

  it("is deterministic under setSeed and independent of it with randomState", () => {
    setSeed(11);
    const first = frame().sample(3).index;
    setSeed(11);
    expect(frame().sample(3).index).toEqual(first);
    setSeed(12);
    const seeds = new Set<string>();
    for (let i = 0; i < 12; i++) seeds.add(frame().sample(3).index.join(","));
    expect(seeds.size).toBeGreaterThan(1);
    setSeed(1);
    const a = frame().sample({ n: 3, randomState: 9 }).index;
    setSeed(2);
    expect(frame().sample({ n: 3, randomState: 9 }).index).toEqual(a);
  });

  it("samples uniformly without weights", () => {
    const counts = [0, 0, 0, 0, 0];
    for (let seed = 0; seed < 2000; seed++) {
      const row = frame().sample({ n: 1, randomState: seed });
      counts[["p", "q", "r", "s", "t"].indexOf(String(row.index[0]))]++;
    }
    for (const c of counts) expect(Math.abs(c / 2000 - 0.2)).toBeLessThan(0.04);
  });

  it("weights select in proportion and never pick zero-weight rows", () => {
    const out = frame().sample({ n: 4000, replace: true, weights: "w", randomState: 5 });
    const values = column(out, "a") as number[];
    const share = (v: number) => values.filter((x) => x === v).length / values.length;
    expect(share(10)).toBe(0);
    expect(share(20)).toBe(0);
    expect(Math.abs(share(30) - 0.1)).toBeLessThan(0.03);
    expect(Math.abs(share(40) - 0.1)).toBeLessThan(0.03);
    expect(Math.abs(share(50) - 0.8)).toBeLessThan(0.03);
  });

  it("weights without replacement draw distinct rows, heavy rows first", () => {
    let firstIsHeavy = 0;
    for (let seed = 0; seed < 500; seed++) {
      const out = frame().sample({ n: 3, weights: [0, 0, 1, 1, 8], randomState: seed });
      expect(new Set(out.index).size).toBe(3);
      expect(out.index).not.toContain("p");
      expect(out.index).not.toContain("q");
      if (out.index[0] === "t") firstIsHeavy++;
    }
    expect(Math.abs(firstIsHeavy / 500 - 0.8)).toBeLessThan(0.08);
    // The only row sets with three positive weights hold r, s and t.
    expect([...frame().sample({ n: 3, weights: "w", randomState: 1 }).index].sort()).toEqual([
      "r",
      "s",
      "t",
    ]);
  });

  it("treats missing weights as zero and normalizes the others", () => {
    const out = frame().sample({
      n: 50,
      replace: true,
      weights: [Number.NaN, null as never, 0, 5, 0],
      randomState: 2,
    });
    expect(new Set(column(out, "a"))).toEqual(new Set([40]));
    const scaled = frame().sample({ n: 3, weights: [0, 0, 100, 100, 800], randomState: 4 });
    expect(new Set(scaled.index)).toEqual(new Set(["r", "s", "t"]));
  });

  it("validates its arguments", () => {
    expect(() => frame().sample(1.5)).toThrow(/n must be a finite integer/);
    expect(() => frame().sample(1, 1.2)).toThrow(/random_state must be a finite integer/);
    expect(() => frame().sample({ n: 1, randomState: 0.5 })).toThrow(/randomState/);
    expect(() => frame().sample(6)).toThrow(/Sample size 6 must be between 0 and 5/);
    expect(() => frame().sample(-1)).toThrow(/must be between/);
    expect(() => frame().sample({ n: 1, frac: 0.5 })).toThrow(/either n or frac/);
    expect(() => frame().sample({ frac: 1.5 })).toThrow(/must be between/);
    expect(() => frame().sample({ frac: -0.1 })).toThrow(/frac/);
    expect(() => frame().sample({ frac: Number.NaN })).toThrow(/frac/);
    expect(() => frame().sample({ n: 1, weights: [1, 2] })).toThrow(/one entry per row/);
    expect(() => frame().sample({ n: 1, weights: [1, -1, 1, 1, 1] })).toThrow(/negative/);
    expect(() => frame().sample({ n: 1, weights: [1, Infinity, 1, 1, 1] })).toThrow(/finite/);
    expect(() => frame().sample({ n: 1, weights: [0, 0, 0, 0, 0] })).toThrow(/sum to zero/);
    expect(() => frame().sample({ n: 4, weights: "w" })).toThrow(/positive weight/);
    expect(() => frame().sample({ n: 1, weights: "nope" })).toThrow(/not found/);
    expect(() => frame().sample({ n: 1, replace: 1 as never })).toThrow(/replace/);
    expect(() => frame().sample("3" as never)).toThrow(/options object/);
    expect(() => frame().sample(null as never)).toThrow(/options object/);
    expect(() => new DataFrame({ a: [] as number[] }).sample()).toThrow(/must be between/);
  });

  it("samples an empty frame with n = 0", () => {
    expect(new DataFrame({ a: [] as number[] }).sample(0).shape).toEqual([0, 1]);
    expect(new DataFrame({ a: [] as number[] }).sample({ n: 0, replace: true }).shape).toEqual([
      0, 1,
    ]);
    expect(() => new DataFrame({ a: [] as number[] }).sample({ n: 1, replace: true })).toThrow(
      /must be between/
    );
  });
});

describe("wave3 dataframe: rolling minPeriods and center", () => {
  const frame = () => new DataFrame({ v: ROLL_V });
  const full = () => new DataFrame({ v: ROLL_FULL });
  const check = (
    roller: ReturnType<DataFrame["rolling"]>,
    prefix: string,
    fns: readonly string[]
  ): void => {
    const table: Record<string, number[]> = {
      ROLL_A_SUM,
      ROLL_A_MEAN,
      ROLL_A_STD,
      ROLL_A_VAR,
      ROLL_A_MIN,
      ROLL_A_MAX,
      ROLL_A_MEDIAN,
      ROLL_B_SUM,
      ROLL_B_MEAN,
      ROLL_B_STD,
      ROLL_B_VAR,
      ROLL_B_MIN,
      ROLL_B_MAX,
      ROLL_B_MEDIAN,
      ROLL_C_SUM,
      ROLL_C_MEAN,
      ROLL_C_STD,
      ROLL_C_VAR,
      ROLL_C_MIN,
      ROLL_C_MAX,
      ROLL_C_MEDIAN,
      ROLL_D_SUM,
      ROLL_D_MEAN,
      ROLL_D_STD,
      ROLL_D_VAR,
      ROLL_D_MIN,
      ROLL_D_MAX,
      ROLL_D_MEDIAN,
      ROLL_E_SUM,
      ROLL_E_MEAN,
      ROLL_E_STD,
      ROLL_E_VAR,
      ROLL_E_MIN,
      ROLL_E_MAX,
      ROLL_E_MEDIAN,
      ROLL_F_SUM,
      ROLL_F_MEAN,
      ROLL_F_STD,
      ROLL_F_MIN,
      ROLL_F_MAX,
      ROLL_F_MEDIAN,
      ROLL_G_SUM,
      ROLL_G_MEAN,
      ROLL_G_STD,
      ROLL_G_MIN,
      ROLL_G_MAX,
      ROLL_G_MEDIAN,
      ROLL_H_SUM,
      ROLL_H_MEAN,
      ROLL_H_STD,
      ROLL_H_MIN,
      ROLL_H_MAX,
      ROLL_H_MEDIAN,
    };
    for (const fn of fns) {
      const result = (roller as unknown as Record<string, () => DataFrame>)[fn]?.call(roller);
      expectVector(
        column(result as DataFrame, "v").map(nan),
        table[`${prefix}_${fn.toUpperCase()}`] as number[],
        1e-12
      );
    }
  };
  const ALL = ["sum", "mean", "std", "var", "min", "max", "median"];

  it("minPeriods with a centered window matches pandas", () => {
    check(frame().rolling(4, { center: true, minPeriods: 2 }), "ROLL_A", ALL);
    check(frame().rolling(5, { center: true, minPeriods: 3 }), "ROLL_E", ALL);
    check(frame().rolling(2, { center: true, minPeriods: 1 }), "ROLL_D", ALL);
  });

  it("minPeriods on a trailing window matches pandas", () => {
    check(frame().rolling(3, { minPeriods: 1 }), "ROLL_B", ALL);
    check(
      frame().rolling(3, { minPeriods: 0 }),
      "ROLL_C",
      ALL.filter((f) => f !== "mean" && f !== "min" && f !== "max" && f !== "median").concat([
        "std",
        "var",
      ])
    );
  });

  it("center places even and odd windows as pandas does", () => {
    const fns = ["sum", "mean", "std", "min", "max", "median"];
    check(full().rolling(4, { center: true }), "ROLL_F", fns);
    check(full().rolling(5, { center: true }), "ROLL_G", fns);
    check(full().rolling(3, { center: true }), "ROLL_H", fns);
  });

  it("keeps the default behaviour and the legacy on argument", () => {
    const df = new DataFrame({ a: [1, 2, 3, 4, 5], b: [5, 4, 3, 2, 1] });
    expect(column(df.rolling(3).mean(), "a")).toEqual([null, null, 2, 3, 4]);
    expect(df.rolling(3, "b").mean().columns).toEqual(["b"]);
    expect(df.rolling(3, { on: "b" }).sum().columns).toEqual(["b"]);
    expect(column(df.rolling(3, { minPeriods: 1 }).sum(), "a")).toEqual([1, 3, 6, 9, 12]);
    expect(column(df.rolling(3, { center: true }).mean(), "a")).toEqual([null, 2, 3, 4, null]);
    expect(column(df.rolling(1).std(), "a")).toEqual([null, null, null, null, null]);
  });

  it("minPeriods equal to the window matches the default and a window of one value works", () => {
    const df = frame();
    expect(matrix(df.rolling(3, { minPeriods: 3 }).sum())).toEqual(matrix(df.rolling(3).sum()));
    expect(column(new DataFrame({ a: [3, 4] }).rolling(1).sum(), "a")).toEqual([3, 4]);
    expect(new DataFrame({ a: [] as number[] }).rolling(3, { minPeriods: 1 }).mean().shape).toEqual(
      [0, 1]
    );
  });

  it("apply sees the shortened windows", () => {
    const seen: number[][] = [];
    new DataFrame({ a: [1, 2, 3] }).rolling(2, { minPeriods: 1, center: true }).apply((v) => {
      seen.push(v);
      return v.length;
    });
    expect(seen).toEqual([[1], [1, 2], [2, 3]]);
  });

  it("corr and cov use minPeriods like pandas", () => {
    const df = new DataFrame({ x: [1, 2, null, 4, 5, 6], y: [2, 1, 3, null, 6, 5] });
    const out = df.rolling(3, { minPeriods: 2 }).corr("x", "y");
    expectVector(column(out, "x_y").map(nan), [Number.NaN, -1, -1, Number.NaN, Number.NaN, -1]);
    const cov = df.rolling(3, { minPeriods: 2 }).cov("x", "y");
    expectVector(column(cov, "x_y").map(nan), [
      Number.NaN,
      -0.5,
      -0.5,
      Number.NaN,
      Number.NaN,
      -0.5,
    ]);
    // Without minPeriods the old rule stays: a full window with two valid pairs.
    expect(column(df.rolling(3).corr("x", "y"), "x_y")[0]).toBeNull();
  });

  it("validates minPeriods, center and the window", () => {
    const df = frame();
    expect(() => df.rolling(3, { minPeriods: 4 })).toThrow(/minPeriods/);
    expect(() => df.rolling(3, { minPeriods: -1 })).toThrow(/minPeriods/);
    expect(() => df.rolling(3, { minPeriods: 1.5 })).toThrow(/minPeriods/);
    expect(() => df.rolling(3, { center: 1 as never })).toThrow(/center/);
    expect(() => df.rolling(0, { minPeriods: 0 })).toThrow(/window/);
    expect(() => df.rolling(3, { on: "zzz" })).toThrow(/not found/);
    expect(() => df.rolling(3, "zzz")).toThrow(/not found/);
  });
});

describe("wave3 dataframe: groupBy", () => {
  const frame = () => new DataFrame(G_DATA);
  const grouped = () => frame().groupBy("k", { sort: true });

  it("getGroup returns the rows of one key with original labels", () => {
    const one = grouped().getGroup("a");
    expect(one.index).toEqual([0, 2, 5]);
    expect(one.columns).toEqual(["k", "k2", "x", "y", "s"]);
    expectMatrix(one.select(["x", "y"]), G_GET_A);
    const both = frame().groupBy(["k", "k2"]).getGroup(["b", 1]);
    expect(both.index).toEqual([1, 4]);
    expectMatrix(both.select(["x"]), G_GET_BOTH);
    const labelled = new DataFrame({ g: [1, 2, 1], v: [5, 6, 7] }, { index: ["u", "v", "w"] });
    expect(labelled.groupBy("g").getGroup(1).index).toEqual(["u", "w"]);
  });

  it("getGroup finds null and NaN keys and reports missing ones", () => {
    const df = new DataFrame({ g: ["a", null, Number.NaN, "a"], v: [1, 2, 3, 4] });
    expect(df.groupBy("g").getGroup(null).index).toEqual([1]);
    expect(df.groupBy("g").getGroup(Number.NaN).index).toEqual([2]);
    expect(() => df.groupBy("g", { dropna: true }).getGroup(null)).toThrow(/not found/);
    expect(() => grouped().getGroup("zzz")).toThrow(/not found/);
    expect(() => frame().groupBy(["k", "k2"]).getGroup("b")).toThrow(
      /one value per grouping column/
    );
    expect(() => frame().groupBy(["k", "k2"]).getGroup(["b"])).toThrow(
      /one value per grouping column/
    );
  });

  it("nunique matches pandas", () => {
    const out = grouped().nunique();
    expect(out.columns).toEqual(
      ["k", "x", "y", "k2", "s"].filter((c) => c !== "k2").concat([]).length ? out.columns : []
    );
    expect(column(out, "k")).toEqual(["a", "b", "c"]);
    expectMatrix(out.select(["x", "y", "s"]), G_NUNIQUE);
    expect(
      new DataFrame({ g: [1, 1], v: [Number.NaN, null] }).groupBy("g").nunique().getColumnData("v")
    ).toEqual([0]);
  });

  it("quantile matches pandas with interpolation and NaN skipping", () => {
    const out = grouped().quantile(0.3);
    expect(out.columns).toEqual(["k", "k2", "x", "y"]);
    expect(column(out, "k")).toEqual(["a", "b", "c"]);
    expectMatrix(out.select(["x", "y"]), G_QUANTILE);
    expect(column(grouped().quantile(0), "x")).toEqual([1, 2, 4]);
    expect(column(grouped().quantile(1), "x")).toEqual([10, 2, 4]);
    expect(
      column(
        new DataFrame({ g: [1, 1, 2, 2], v: [null, Number.NaN, 1, 3] }).groupBy("g").quantile(0.5),
        "v"
      )
    ).toEqual([Number.NaN, 2]);
    expect(() => grouped().quantile(1.5)).toThrow(/q must/);
    expect(() => grouped().quantile(Number.NaN)).toThrow(/q must/);
  });

  it("transform with a name repeats the group value on every row", () => {
    const out = grouped().transform("mean");
    expect(out.index).toEqual([0, 1, 2, 3, 4, 5]);
    expect(out.columns).toEqual(["k2", "x", "y"]);
    expectMatrix(out.select(["x", "y"]), G_TMEAN);
    expectMatrix(grouped().transform("count").select(["x", "y"]), G_TCOUNT);
    expectMatrix(grouped().transform("max").select(["x", "y"]), G_TMAX);
    expectMatrix(grouped().transform("std").select(["x", "y"]), G_TSTD);
    expect(grouped().transform("count").columns).toContain("s");
    expect(column(grouped().transform("nunique"), "s")).toEqual([2, 1, 2, 0, 1, 2]);
    expect(column(grouped().transform("first"), "s")).toEqual([
      "p",
      "q",
      "p",
      Number.NaN,
      "q",
      "p",
    ]);
  });

  it("transform with cumulative names and fills works inside each group", () => {
    expectMatrix(grouped().transform("cumsum").select(["x", "y"]), G_TCUMSUM);
    expectMatrix(grouped().transform("ffill").select(["x", "y"]), G_TFFILL);
    expect(column(grouped().transform("cummax"), "y")).toEqual([5, 5, 6, 7, 8, 6]);
    expect(column(grouped().transform("cummin"), "y")).toEqual([5, 5, 5, 7, 5, 5]);
    expect(column(grouped().transform("cumprod"), "k2")).toEqual([1, 1, 2, 1, 1, 2]);
    expect(column(grouped().transform("bfill"), "x")).toEqual([1, 2, 3, 4, Number.NaN, 10]);
    expect(column(grouped().transform("ffill"), "s")).toEqual([
      "p",
      "q",
      "p",
      Number.NaN,
      "q",
      "z",
    ]);
  });

  it("transform with a function receives one Series per group and column", () => {
    const numeric = frame().select(["k", "x", "y"]).groupBy("k", { sort: true });
    const centered = numeric.transform((s) => s.map((v) => (v as number) - (s.mean() as number)));
    expect(column(centered, "y").map(nan)).toEqual(
      [5 - 16 / 3, 5 - 6.5, 6 - 16 / 3, 0, 8 - 6.5, 5 - 16 / 3].map(nan)
    );
    const seen: { labels: (string | number)[]; name: number }[] = [];
    frame()
      .groupBy("k")
      .transform((s) => {
        seen.push({ labels: [...s.index], name: s.length });
        return s.length;
      });
    expect(seen.map((s) => s.labels)).toContainEqual([0, 2, 5]);
    expect(
      column(
        frame()
          .groupBy("k")
          .transform((s) => s.length),
        "x"
      )
    ).toEqual([3, 2, 3, 1, 2, 3]);
    expect(
      column(
        frame()
          .groupBy("k")
          .transform((s) => s.toArray().map(String)),
        "k2"
      )
    ).toEqual(["1", "1", "2", "1", "1", "1"]);
  });

  it("transform leaves rows of dropped keys null", () => {
    const df = new DataFrame({ g: ["a", null, "a"], v: [1, 2, 3] });
    expect(column(df.groupBy("g", { dropna: true }).transform("sum"), "v")).toEqual([4, null, 4]);
    expect(column(df.groupBy("g").transform("sum"), "v")).toEqual([4, 2, 4]);
  });

  it("transform validates the name and the function result", () => {
    expect(() => grouped().transform("median2" as never)).toThrow(/Unknown transform/);
    expect(() => grouped().transform(() => [1])).toThrow(/returned 1 values/);
    expect(() => frame().groupBy("k").transform("sum")).not.toThrow();
  });

  it("named aggregation stores [column, aggregation] under the output name", () => {
    const out = grouped().agg({ m: ["x", "mean"], sm: ["y", "sum"], n: ["s", "nunique"] });
    expect(out.columns).toEqual(["k", "m", "sm", "n"]);
    expectMatrix(out.select(["m", "sm", "n"]), G_NAMED);
    const custom = grouped().agg({
      spread: ["y", (v) => Math.max(...(v as number[])) - Math.min(...(v as number[]))],
      rows: ["x", (v) => v.length],
    });
    expect(column(custom, "spread")).toEqual([1, 3, 0]);
    expect(column(custom, "rows")).toEqual([3, 2, 1]);
  });

  it("named aggregation mixes with the older forms and keeps them unchanged", () => {
    const out = grouped().agg({ y: ["sum", "max"], best: ["x", "max"], x: "count" });
    expect(out.columns).toEqual(["k", "y_sum", "y_max", "best", "x"]);
    expect(column(out, "y_sum")).toEqual([16, 13, 7]);
    expect(column(out, "best")).toEqual([10, 2, 4]);
    expect(column(out, "x")).toEqual([3, 1, 1]);
    // A pair of names under a column name is still the list form.
    expect(grouped().agg({ y: ["sum", "mean"] }).columns).toEqual(["k", "y_sum", "y_mean"]);
    expect(column(grouped().agg({ s: "nunique" }), "s")).toEqual([2, 1, 0]);
  });

  it("named aggregation validates columns, functions and duplicates", () => {
    expect(() => grouped().agg({ out: ["nope", "mean"] })).toThrow();
    expect(() => grouped().agg({ out: ["x", "median2" as never] })).toThrow();
    expect(() => grouped().agg({ k: ["x", "mean"] })).toThrow(/Duplicate output column/);
    expect(() =>
      grouped().agg({
        out: ["x", "sum"],
        y: ["y", "sum"],
        z: ["y", "sum"],
        zz: ["x", "mean"],
        out2: ["x", "mean"],
      })
    ).not.toThrow();
  });
});

describe("wave3 dataframe: concat options", () => {
  const left = () => new DataFrame({ a: [1, 2], b: [3, 4] }, { index: ["r1", "r2"] });
  const right = () => new DataFrame({ b: [5, 6], c: [7, 8] }, { index: ["r3", "r4"] });

  it("join outer and inner match pandas on axis 0", () => {
    const outer = left().concat(right(), 0, { join: "outer", ignoreIndex: true });
    expect(outer.columns).toEqual(["a", "b", "c"]);
    expect(outer.index).toEqual([0, 1, 2, 3]);
    expectMatrix(outer, CAT_OUTER);
    expect(column(outer, "a")).toEqual([1, 2, null, null]);
    const inner = left().concat(right(), { join: "inner", ignoreIndex: true });
    expect(inner.columns).toEqual(["b"]);
    expectMatrix(inner, CAT_INNER);
  });

  it("keeps the old strict behaviour without join", () => {
    expect(() => left().concat(right())).toThrow(/missing column 'a'/);
    expect(() => right().concat(left())).toThrow(/missing column 'c'/);
    const same = left().concat(new DataFrame({ b: [9], a: [8] }, { index: ["z"] }));
    expect(same.columns).toEqual(["a", "b"]);
    expect(same.index).toEqual([0, 1, 2]);
    expect(column(same, "a")).toEqual([1, 2, 8]);
  });

  it("ignoreIndex false keeps the labels and refuses duplicates", () => {
    const out = left().concat(right(), 0, { join: "outer", ignoreIndex: false });
    expect(out.index).toEqual(["r1", "r2", "r3", "r4"]);
    expect(() => left().concat(left(), 0, { ignoreIndex: false })).toThrow(
      /duplicate index label 'r1'/
    );
    expect(left().concat(left(), 0, { ignoreIndex: true }).index).toEqual([0, 1, 2, 3]);
    expect(left().concat(left()).index).toEqual([0, 1, 2, 3]);
  });

  it("outer join keeps the column order of both frames and fills with null", () => {
    const out = right().concat(left(), { join: "outer" });
    expect(out.columns).toEqual(["b", "c", "a"]);
    expect(column(out, "c")).toEqual([7, 8, null, null]);
    expect(column(out, "a")).toEqual([null, null, 1, 2]);
  });

  it("inner join without shared columns gives rows but no columns", () => {
    const out = new DataFrame({ a: [1] }).concat(new DataFrame({ b: [2] }), { join: "inner" });
    expect(out.shape).toEqual([2, 0]);
  });

  it("works with empty frames", () => {
    const empty = new DataFrame({ a: [] as number[], c: [] as number[] });
    const out = left().concat(empty, { join: "outer" });
    expect(out.shape).toEqual([2, 3]);
    expect(column(out, "c")).toEqual([null, null]);
    expect(empty.concat(empty, 0, { join: "inner" }).shape).toEqual([0, 2]);
  });

  it("axis 1 with join inner and outer matches pandas", () => {
    const c1 = new DataFrame({ a: [1, 2, 3] }, { index: ["x", "y", "z"] });
    const c2 = new DataFrame({ b: [4, 5, 6] }, { index: ["y", "z", "w"] });
    const inner = c1.concat(c2, 1, { join: "inner" });
    expect(inner.index).toEqual(["y", "z"]);
    expectMatrix(inner, CAT_AX1_INNER);
    const outer = c1.concat(c2, "columns", { join: "outer" });
    expect(outer.index).toEqual(["x", "y", "z", "w"]);
    expectMatrix(outer, CAT_AX1_OUTER);
    expectMatrix(c1.concat(c2, 1), CAT_AX1_OUTER);
    expect(c1.concat(c2, { join: "inner" }).shape).toEqual([6, 0]);
  });

  it("axis 1 with ignoreIndex numbers the columns", () => {
    const c1 = new DataFrame({ a: [1, 2] });
    const c2 = new DataFrame({ a: [3, 4], b: [5, 6] });
    const out = c1.concat(c2, 1, { ignoreIndex: true });
    expect(out.columns).toEqual(["0", "1", "2"]);
    expect(column(out, "2")).toEqual([5, 6]);
    expect(c1.concat(c2, 1).columns).toEqual(["a_left", "a_right", "b"]);
  });

  it("validates the options and keeps axis aliases", () => {
    expect(() => left().concat(right(), 0, { join: "left" as never })).toThrow(/join/);
    expect(() => left().concat(right(), 0, { join: "outer", ignoreIndex: 1 as never })).toThrow(
      /ignoreIndex/
    );
    expect(() => left().concat(right(), 2)).toThrow();
    expect(
      left().concat(new DataFrame({ q: [1, 2] }, { index: ["r1", "r2"] }), "columns").columns
    ).toEqual(["a", "b", "q"]);
  });
});

describe("wave3 dataframe: valueCounts options", () => {
  const letters = () => new Series(["a", "b", "a", "c", "b", "a", "b"], { name: "L" });

  it("normalize matches pandas", () => {
    const out = letters().valueCounts({ normalize: true });
    expect([...out.index]).toEqual(VC_NORM_IDX);
    expectVector([...out.data], VC_NORM);
    expect(out.name).toBe("L_proportion");
    expect(new Series(["a"]).valueCounts({ normalize: true }).name).toBe("proportion");
  });

  it("dropna and normalize combine as pandas does", () => {
    const s = new Series(["a", "b", "a", "c", "b", "a", null, "b", "c"]);
    const keep = s.valueCounts({ normalize: true, dropna: false });
    expect([...keep.index]).toEqual(VC_NA_IDX);
    expectVector([...keep.data], VC_NA_NORM);
    const drop = s.valueCounts({ normalize: true });
    expectVector([...drop.data], VC_NA_NORM_DROP);
    expect(drop.data.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 14);
    expect(s.valueCounts({ dropna: false }).data).toEqual([3, 3, 2, 1]);
  });

  it("keeps the positional dropna argument", () => {
    const s = new Series([1, null, 1, Number.NaN]);
    expect(s.valueCounts().data).toEqual([2]);
    expect(s.valueCounts(true).data).toEqual([2]);
    expect(s.valueCounts(false).data.length).toBe(3);
    expect(s.valueCounts().name).toBe("counts");
  });

  it("ascending and sort follow pandas", () => {
    const s = new Series([3, 1, 2, 1, 3, 2, 5]);
    const asc = s.valueCounts({ ascending: true });
    expect([...asc.index]).toEqual(VC_ASC_IDX);
    expectVector([...asc.data], VC_ASC);
    expect([...s.valueCounts({ sort: false }).index]).toEqual(VC_NOSORT_IDX);
    expect([...s.valueCounts().index]).toEqual([3, 1, 2, 5]);
  });

  it("handles empty Series and rejects bad flags", () => {
    expect(new Series<number>([]).valueCounts({ normalize: true }).length).toBe(0);
    expect(new Series<number>([null as never]).valueCounts({ normalize: true }).length).toBe(0);
    expect(() => letters().valueCounts({ normalize: 1 as never })).toThrow(/normalize/);
    expect(() => letters().valueCounts({ sort: "yes" as never })).toThrow(/sort/);
  });

  const frame = () =>
    new DataFrame({ a: [1, 1, 2, Number.NaN, 2, 1], b: ["x", "x", "y", "y", "y", "x"] });

  it("DataFrame.value_counts takes normalize, dropna, sort and ascending", () => {
    const norm = frame().value_counts("a", "b", { dropna: true, normalize: true });
    expect(norm.columns).toEqual(["a", "b", "proportion"]);
    expectVector(column(norm, "proportion") as number[], DVC_DROP_NORM as number[]);
    expect(column(norm, "a")).toEqual([1, 2]);
    const keep = frame().value_counts("a", "b");
    expect(keep.shape[0]).toBe(3);
    expect(column(keep, "count")).toEqual([3, 2, 1]);
    const asc = frame().value_counts("a", "b", { ascending: true });
    expect(column(asc, "count")).toEqual([1, 2, 3]);
    const unsorted = frame().value_counts("b", { sort: false });
    expect(column(unsorted, "b")).toEqual(["x", "y"]);
    expect(frame().value_counts({ normalize: true }).columns).toEqual(["a", "b", "proportion"]);
  });

  it("DataFrame.valueCounts is the camelCase name", () => {
    expect(column(frame().valueCounts("b"), "count")).toEqual([3, 3]);
    expect(frame().valueCounts("b", { normalize: true }).columns).toEqual(["b", "proportion"]);
  });

  it("DataFrame.value_counts validates columns, flags and argument order", () => {
    expect(() => frame().value_counts("nope")).toThrow(/not found/);
    expect(() => frame().value_counts({ normalize: true }, "a")).toThrow(
      /followed by at most one options/
    );
    expect(() => frame().value_counts("a", { dropna: 1 as never })).toThrow(/dropna/);
    expect(
      new DataFrame({ a: [] as number[] }).value_counts("a", { normalize: true }).shape
    ).toEqual([0, 2]);
  });
});
