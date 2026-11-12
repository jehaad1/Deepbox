import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { tensor } from "../../src/ndarray";
import { choice, permutation, setSeed, shuffle } from "../../src/random";

// Semantics follow numpy.random.Generator.shuffle / permutation / choice with the axis argument:
// whole slices along the axis move together, and choice returns
// shape[:axis] + size + shape[axis + 1:] (numpy 2.x checked for the output shapes).

const M = (): ReturnType<typeof tensor> =>
  tensor(
    [
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ],
    { dtype: "float64" }
  );

function rowsOf(t: { toArray(): unknown }): number[][] {
  return t.toArray() as number[][];
}

describe("random axis option", () => {
  it("shuffle with axis 0 permutes whole rows in place", () => {
    for (let seed = 0; seed < 20; seed++) {
      setSeed(seed);
      const x = M();
      shuffle(x, { axis: 0 });
      const rows = rowsOf(x);
      const sorted = [...rows].sort((p, q) => (p[0] as number) - (q[0] as number));
      expect(sorted).toEqual(rowsOf(M()));
    }
  });

  it("shuffle with axis 1 applies one column permutation to every row", () => {
    for (let seed = 0; seed < 20; seed++) {
      setSeed(seed);
      const x = tensor(
        [
          [1, 2, 3],
          [10, 20, 30],
        ],
        { dtype: "float64" }
      );
      shuffle(x, { axis: 1 });
      const rows = rowsOf(x);
      const first = rows[0] as number[];
      expect(rows[1]).toEqual(first.map((v) => v * 10));
      expect([...first].sort()).toEqual([1, 2, 3]);
    }
  });

  it("shuffle accepts a negative axis and rejects an out-of-range axis", () => {
    setSeed(1);
    const x = M();
    shuffle(x, { axis: -2 });
    expect(() => shuffle(M(), { axis: 2 })).toThrow(InvalidParameterError);
    expect(() => shuffle(M(), { axis: 0.5 })).toThrow(InvalidParameterError);
  });

  it("permutation with axis returns a shuffled copy and keeps the input", () => {
    setSeed(3);
    const x = M();
    const y = permutation(x, { axis: 0 });
    expect(rowsOf(x)).toEqual(rowsOf(M()));
    expect(y.shape).toEqual([4, 2]);
    const sorted = rowsOf(y).sort((p, q) => (p[0] as number) - (q[0] as number));
    expect(sorted).toEqual(rowsOf(M()));
    for (const r of rowsOf(y)) expect((r[1] as number) - (r[0] as number)).toBe(1);
  });

  it("permutation of an integer ignores the axis like numpy", () => {
    setSeed(3);
    expect(permutation(5, { axis: 0 }).size).toBe(5);
  });

  it("choice with axis 0 samples rows (numpy: Generator.choice(a, 3, axis=0))", () => {
    setSeed(5);
    const out = choice(M(), 3, false, undefined, { axis: 0 });
    expect(out.shape).toEqual([3, 2]);
    const rows = rowsOf(out);
    for (const r of rows) expect((r[1] as number) - (r[0] as number)).toBe(1);
    expect(new Set(rows.map((r) => r[0])).size).toBe(3);
  });

  it("choice returns shape[:axis] + size + shape[axis + 1:]", () => {
    const a = tensor(
      Array.from({ length: 2 }, (_, i) =>
        Array.from({ length: 3 }, (_, j) => Array.from({ length: 4 }, (_, k) => i * 12 + j * 4 + k))
      ),
      { dtype: "float64" }
    );
    // numpy: g.choice(a, (2, 2), axis=1).shape == (2, 2, 2, 4); g.choice(a, 2, axis=-1).shape == (2, 3, 2)
    expect(choice(a, [2, 2], true, undefined, { axis: 1 }).shape).toEqual([2, 2, 2, 4]);
    expect(choice(a, 2, true, undefined, { axis: -1 }).shape).toEqual([2, 3, 2]);
    expect(choice(a, 2, false, undefined, { axis: 2 }).shape).toEqual([2, 3, 2]);
  });

  it("choice with axis honors probabilities (numpy: p=[0, 0, 1, 0] always picks row 2)", () => {
    setSeed(2);
    const out = choice(M(), 3, true, tensor([0, 0, 1, 0]), { axis: 0 });
    expect(rowsOf(out)).toEqual([
      [5, 6],
      [5, 6],
      [5, 6],
    ]);
  });

  it("choice with axis validates the axis and the population size", () => {
    expect(() => choice(M(), 1, true, undefined, { axis: 2 })).toThrow(InvalidParameterError);
    expect(() => choice(M(), 5, false, undefined, { axis: 0 })).toThrow(InvalidParameterError);
  });

  it("the default still flattens N-D input", () => {
    setSeed(1);
    expect(choice(M(), 3).shape).toEqual([3]);
    const x = M();
    shuffle(x);
    expect(x.shape).toEqual([4, 2]);
  });
});
