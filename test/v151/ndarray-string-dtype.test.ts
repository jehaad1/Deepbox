import { describe, expect, it } from "vitest";
import { DTypeError } from "../../src/core";
import {
  argsort,
  booleanIndex,
  equal,
  fancyIndex,
  flip,
  intersect1d,
  notEqual,
  roll,
  setdiff1d,
  sort,
  tensor,
  union1d,
  unique,
  where,
} from "../../src/ndarray";

// Reference: numpy 2.x on '<U' arrays. numpy orders strings by Unicode code point.

const s = (data: string[] | string[][]) => tensor(data);

describe("string dtype operations", () => {
  it("equal and notEqual compare elements and broadcast", () => {
    expect(equal(s(["a", "b"]), s(["a", "c"])).toArray()).toEqual([1, 0]);
    expect(notEqual(s(["a", "b"]), s(["a"])).toArray()).toEqual([0, 1]);
    expect(notEqual(s(["a", "b"]), tensor("a")).toArray()).toEqual([0, 1]);
  });

  it("equal rejects a string against a number", () => {
    expect(() => equal(s(["a"]), tensor([1]))).toThrow(DTypeError);
  });

  it("where selects strings", () => {
    const cond = tensor([true, false, true]);
    expect(where(cond, s(["x", "y", "z"]), s(["p", "q", "r"])).toArray()).toEqual(["x", "q", "z"]);
  });

  it("where rejects mixed string and numeric operands", () => {
    expect(() => where(tensor([true]), s(["x"]), tensor([1]))).toThrow(DTypeError);
  });

  it("sort and argsort use code point order and are stable", () => {
    const a = s(["b", "a", "B", "é", "b"]);
    expect(sort(a).toArray()).toEqual(["B", "a", "b", "b", "é"]);
    expect(argsort(a).toArray()).toEqual([2, 1, 0, 4, 3]);
    expect(sort(a, -1, true).toArray()).toEqual(["é", "b", "b", "a", "B"]);
    // descending argsort keeps the original order of equal elements
    expect(argsort(a, -1, true).toArray()).toEqual([3, 0, 4, 1, 2]);
  });

  it("sorts by code point, not by UTF-16 code unit", () => {
    // numpy.sort(['😀', '～']) -> ['～', '😀']
    expect(sort(s(["😀", "～"])).toArray()).toEqual(["～", "😀"]);
  });

  it("sorts along an axis", () => {
    expect(
      sort(
        s([
          ["b", "a"],
          ["d", "c"],
        ]),
        0
      ).toArray()
    ).toEqual([
      ["b", "a"],
      ["d", "c"],
    ]);
    expect(
      sort(
        s([
          ["b", "a"],
          ["d", "c"],
        ]),
        1
      ).toArray()
    ).toEqual([
      ["a", "b"],
      ["c", "d"],
    ]);
  });

  it("unique returns sorted values with index, inverse and counts", () => {
    const r = unique(s(["b", "a", "B", "é", "b"]), {
      returnIndex: true,
      returnInverse: true,
      returnCounts: true,
    });
    expect(r.values.toArray()).toEqual(["B", "a", "b", "é"]);
    expect(r.indices?.toArray()).toEqual([2, 1, 0, 3]);
    expect(r.inverse?.toArray()).toEqual([2, 1, 0, 3, 2]);
    expect(r.counts?.toArray()).toEqual([1, 1, 2, 1]);
  });

  it("unique along an axis compares rows", () => {
    const r = unique(
      s([
        ["b", "a"],
        ["b", "a"],
        ["a", "z"],
      ]),
      { axis: 0 }
    );
    expect(r.values.toArray()).toEqual([
      ["a", "z"],
      ["b", "a"],
    ]);
  });

  it("flip and roll move strings", () => {
    expect(
      flip(
        s([
          ["a", "b"],
          ["c", "d"],
        ]),
        0
      ).toArray()
    ).toEqual([
      ["c", "d"],
      ["a", "b"],
    ]);
    expect(
      roll(
        s([
          ["a", "b", "c"],
          ["d", "e", "f"],
        ]),
        1,
        1
      ).toArray()
    ).toEqual([
      ["c", "a", "b"],
      ["f", "d", "e"],
    ]);
  });

  it("booleanIndex and fancyIndex select strings", () => {
    expect(booleanIndex(s(["a", "b", "c"]), tensor([true, false, true])).toArray()).toEqual([
      "a",
      "c",
    ]);
    expect(fancyIndex(s(["a", "b", "c"]), tensor([2, 0], { dtype: "int32" })).toArray()).toEqual([
      "c",
      "a",
    ]);
  });

  it("set operations work on strings", () => {
    expect(union1d(s(["b", "a"]), s(["c", "a"])).toArray()).toEqual(["a", "b", "c"]);
    expect(intersect1d(s(["b", "a", "a"]), s(["c", "a"])).toArray()).toEqual(["a"]);
    expect(setdiff1d(s(["b", "a", "a"]), s(["a"])).toArray()).toEqual(["b"]);
  });

  it("works on strided string views", () => {
    const m = s([
      ["d", "x"],
      ["a", "y"],
      ["c", "z"],
    ]);
    const col = m.slice({}, 0);
    expect(sort(col).toArray()).toEqual(["a", "c", "d"]);
    expect(flip(col).toArray()).toEqual(["c", "a", "d"]);
  });
});
