import { describe, expect, it } from "vitest";
import { DataValidationError, InvalidParameterError } from "../../src/core/errors/index";
import { DataFrame } from "../../src/dataframe/index";

const col = (df: DataFrame, name = "a"): unknown[] => df.get(name).toArray();

// pandas 3.1: pd.to_numeric(pd.Series(['1', '2.5', ' 3 ', '1e3', 'abc', '0x10', None]), errors='coerce')
//   -> [1, 2.5, 3, 1000, nan, nan, nan]; errors='raise' raises ValueError on 'abc'.
describe("astype number conversion is strict decimal (DF-23)", () => {
  const df = new DataFrame({
    a: ["1", "2.5", " 3 ", "1e3", "abc", "0x10", "0b11", "", null, "Infinity", "-.5"],
  });

  it("coerce (the default) gives NaN for text that is not a decimal number", () => {
    const out = col(df.astype("a", "number"));
    expect(out.slice(0, 4)).toEqual([1, 2.5, 3, 1000]);
    expect(out[4]).toBeNaN();
    expect(out[5]).toBeNaN();
    expect(out[6]).toBeNaN();
    expect(out[7]).toBeNaN();
    expect(out[8]).toBeNaN();
    expect(out[9]).toBe(Infinity);
    expect(out[10]).toBe(-0.5);
    expect(col(df.astype("a", "number", { errors: "coerce" }))[5]).toBeNaN();
  });

  it("raise throws a DataValidationError naming the value and column", () => {
    expect(() => df.astype("a", "number", { errors: "raise" })).toThrow(DataValidationError);
    expect(() => df.astype({ a: "number" }, { errors: "raise" })).toThrow(/"abc" in column 'a'/);
    const clean = new DataFrame({ a: ["1", " 2 ", null, ""] });
    expect(col(clean.astype("a", "number", { errors: "raise" }))).toEqual([
      1,
      2,
      Number.NaN,
      Number.NaN,
    ]);
  });
});

describe("astype boolean conversion of strings (DF-23)", () => {
  const df = new DataFrame({
    a: ["true", "FALSE", "1", "0", "Yes", "no", " Y ", "maybe", "", null],
  });

  it("understands true/false/1/0/yes/no in any case and gives null otherwise", () => {
    expect(col(df.astype("a", "boolean"))).toEqual([
      true,
      false,
      true,
      false,
      true,
      false,
      null,
      null,
      null,
      null,
    ]);
  });

  it("raise throws for text it cannot read", () => {
    expect(() => df.astype("a", "boolean", { errors: "raise" })).toThrow(/cannot convert " Y "/);
    const clean = new DataFrame({ a: ["YES", "no", null] });
    expect(col(clean.astype("a", "boolean", { errors: "raise" }))).toEqual([true, false, null]);
  });

  it("non-string values still follow truthiness", () => {
    expect(col(new DataFrame({ a: [0, 2, true, false] }).astype("a", "boolean"))).toEqual([
      false,
      true,
      true,
      false,
    ]);
  });
});

describe("astype options validation", () => {
  it("rejects an unknown errors value", () => {
    const df = new DataFrame({ a: ["1"] });
    expect(() => df.astype("a", "number", { errors: "ignore" as unknown as "raise" })).toThrow(
      InvalidParameterError
    );
  });
});
