import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core/errors/index";
import { DataFrame } from "../../src/dataframe/index";

const rows = (df: DataFrame): Record<string, unknown[]> => {
  const out: Record<string, unknown[]> = {};
  for (const c of df.columns) out[c] = df.get(c).toArray();
  return out;
};

const left = () => new DataFrame({ a: [1, 1, 2, 3], b: ["x", "y", "x", "z"], v: [10, 20, 30, 40] });
const right = () => new DataFrame({ a: [1, 1, 2, 4], b: ["x", "x", "y", "w"], v: [1, 2, 3, 4] });

// pandas 3.1: left.merge(right, on=['a', 'b'], how=...) with the frames above.
describe("merge and join with several key columns (DF-12)", () => {
  it("inner merge on two keys", () => {
    expect(rows(left().merge(right(), { on: ["a", "b"] }))).toEqual({
      a: [1, 1],
      b: ["x", "x"],
      v_x: [10, 10],
      v_y: [1, 2],
    });
  });

  it("left merge on two keys keeps unmatched left rows", () => {
    expect(rows(left().merge(right(), { on: ["a", "b"], how: "left" }))).toEqual({
      a: [1, 1, 1, 2, 3],
      b: ["x", "x", "y", "x", "z"],
      v_x: [10, 10, 20, 30, 40],
      v_y: [1, 2, null, null, null],
    });
  });

  it("outer merge fills the shared key columns from the right side", () => {
    const out = rows(left().merge(right(), { on: ["a", "b"], how: "outer" }));
    // Deepbox lists the left-driven rows first, then the unmatched right rows.
    expect(out.a).toEqual([1, 1, 1, 2, 3, 2, 4]);
    expect(out.b).toEqual(["x", "x", "y", "x", "z", "y", "w"]);
    expect(out.v_x).toEqual([10, 10, 20, 30, 40, null, null]);
    expect(out.v_y).toEqual([1, 2, null, null, null, 3, 4]);
  });

  it("leftOn and rightOn accept arrays and keep both key columns", () => {
    const r = right().rename({ a: "a2", b: "b2" });
    const out = rows(left().merge(r, { leftOn: ["a", "b"], rightOn: ["a2", "b2"] }));
    expect(out).toEqual({
      a: [1, 1],
      b: ["x", "x"],
      v_x: [10, 10],
      a2: [1, 1],
      b2: ["x", "x"],
      v_y: [1, 2],
    });
  });

  it("join accepts an array of key columns", () => {
    const l = new DataFrame({ a: [1, 2], b: ["x", "y"], p: [1, 2] });
    const r = new DataFrame({ a: [1, 2], b: ["x", "z"], q: [5, 6] });
    expect(rows(l.join(r, ["a", "b"]))).toEqual({ a: [1], b: ["x"], p: [1], q: [5] });
  });

  it("rows with a null part in any key never match", () => {
    const l = new DataFrame({ a: [1, null], b: ["x", "x"], p: [1, 2] });
    const r = new DataFrame({ a: [1, null], b: ["x", "x"], q: [5, 6] });
    expect(rows(l.merge(r, { on: ["a", "b"] }))).toEqual({ a: [1], b: ["x"], p: [1], q: [5] });
  });

  it("a single name still works as before", () => {
    expect(rows(left().merge(right(), { on: "a" })).a).toEqual([1, 1, 1, 1, 2]);
  });
});

describe("merge and join validate the key option (DF-12)", () => {
  it("rejects an empty string, an empty array and a repeated name", () => {
    for (const bad of ["", [], [""], ["a", "a"]]) {
      expect(() => left().merge(right(), { on: bad })).toThrow(InvalidParameterError);
      expect(() => left().join(right(), bad)).toThrow(InvalidParameterError);
    }
    expect(() => left().merge(right(), { on: "" })).toThrow(
      /on must not contain an empty column name/
    );
    expect(() => left().merge(right(), { on: [""] })).toThrow(/empty column name/);
  });

  it("rejects leftOn and rightOn of different lengths or with an empty name", () => {
    expect(() => left().merge(right(), { leftOn: ["a", "b"], rightOn: ["a"] })).toThrow(
      /same number of columns/
    );
    expect(() => left().merge(right(), { leftOn: "", rightOn: "a" })).toThrow(
      InvalidParameterError
    );
  });

  it("names the missing column", () => {
    expect(() => left().merge(right(), { on: ["a", "nope"] })).toThrow(
      "Column 'nope' not found in left DataFrame"
    );
    expect(() => left().join(right(), ["a", "nope"])).toThrow(
      "Join column 'nope' not found in left DataFrame"
    );
  });
});

describe("concat accepts a DataFrame or an array of DataFrames (DOC-3)", () => {
  const a = () => new DataFrame({ x: [1, 2], y: ["a", "b"] });
  const b = () => new DataFrame({ x: [3], y: ["c"] });
  const c = () => new DataFrame({ x: [4, 5], y: ["d", "e"] });

  it("stacks the rows of every frame in the array", () => {
    const out = a().concat([b(), c()]);
    expect(rows(out)).toEqual({ x: [1, 2, 3, 4, 5], y: ["a", "b", "c", "d", "e"] });
    expect(out.index).toEqual([0, 1, 2, 3, 4]);
  });

  it("matches the single-frame form for a one-element array", () => {
    expect(rows(a().concat([b()]))).toEqual(rows(a().concat(b())));
  });

  it("passes axis and options on to every step", () => {
    const out = a().concat([new DataFrame({ z: [9, 8] })], 1);
    expect(out.columns).toEqual(["x", "y", "z"]);
    const wide = a().concat([new DataFrame({ x: [3], q: [1] })], { join: "outer" });
    expect(wide.columns).toEqual(["x", "y", "q"]);
  });

  it("an empty array returns an equal DataFrame", () => {
    expect(rows(a().concat([]))).toEqual(rows(a()));
  });

  it("throws InvalidParameterError for anything else", () => {
    for (const bad of [null, undefined, 5, "x", {}, [1], [b(), null]]) {
      expect(() => a().concat(bad as unknown as DataFrame)).toThrow(InvalidParameterError);
    }
  });
});
