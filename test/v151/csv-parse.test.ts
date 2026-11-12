import { describe, expect, it } from "vitest";
import { DataFrame } from "../../src/dataframe/index";

const col = (df: DataFrame, name: string): unknown[] => df.get(name).toArray();

// Expected values come from pandas 3.1 read_csv on the same text unless noted.
describe("fromCsvString quote handling (DF-3)", () => {
  it("keeps a quote in the middle of an unquoted field literally", () => {
    // pandas: name = ['pipe 12" steel', 'say "hi" there'], n = [5, 2]
    const df = DataFrame.fromCsvString('name,n\npipe 12" steel,5\nsay "hi" there,2\n');
    expect(col(df, "name")).toEqual(['pipe 12" steel', 'say "hi" there']);
    expect(col(df, "n")).toEqual([5, 2]);
  });

  it("closes a quoted field and appends trailing characters literally", () => {
    // pandas: a = ['xy']
    const df = DataFrame.fromCsvString('a,b\n"x"y,1\n');
    expect(col(df, "a")).toEqual(["xy"]);
  });

  it("still handles doubled quotes and delimiters inside quoted fields", () => {
    const df = DataFrame.fromCsvString('a,b\n"he said ""hi"", ok",2\n"",3\n');
    expect(col(df, "a")).toEqual(['he said "hi", ok', null]);
  });
});

describe("fromCsvString number inference (DF-4, DF-11)", () => {
  it("keeps integers beyond 2^53 as the original text", () => {
    // pandas keeps them as int64; JavaScript numbers cannot, so the text is kept.
    const df = DataFrame.fromCsvString(
      "a,b\n9007199254740993,9007199254740991\n-9007199254740993,1"
    );
    expect(col(df, "a")).toEqual(["9007199254740993", "-9007199254740993"]);
    expect(col(df, "b")).toEqual([9007199254740991, 1]);
  });

  it("applies the leading-zero text rule after the sign", () => {
    const df = DataFrame.fromCsvString("a,b,c,d\n-007,+007,-0.5,0\n");
    expect(col(df, "a")).toEqual(["-007"]);
    expect(col(df, "b")).toEqual(["+007"]);
    expect(col(df, "c")).toEqual([-0.5]);
    expect(col(df, "d")).toEqual([0]);
  });

  it("keeps non-finite parses as text but still reads Infinity words", () => {
    const df = DataFrame.fromCsvString("a,b,c\n1e400,Infinity,-Infinity\n");
    expect(col(df, "a")).toEqual(["1e400"]);
    expect(col(df, "b")).toEqual([Infinity]);
    expect(col(df, "c")).toEqual([-Infinity]);
  });
});

describe("fromCsvString missing values, booleans, headers and short rows (DF-18)", () => {
  it("treats the pandas missing tokens as null by default and NaN as NaN", () => {
    const df = DataFrame.fromCsvString("a,b\nNA,N/A\nNULL,None\nn/a,#N/A\n<NA>,nan\n");
    expect(col(df, "a")).toEqual([null, null, null, null]);
    expect(col(df, "b")).toEqual([null, null, null, NaN]);
  });

  it("naValues replaces the default token set; empty cells stay null", () => {
    const df = DataFrame.fromCsvString("a,b\nNA,missing\n,x\n", { naValues: ["missing"] });
    expect(col(df, "a")).toEqual(["NA", null]);
    expect(col(df, "b")).toEqual([null, "x"]);
    const none = DataFrame.fromCsvString("a\nNA\nnull\n", { naValues: [] });
    expect(col(none, "a")).toEqual(["NA", "null"]);
  });

  it("rejects an invalid naValues", () => {
    expect(() =>
      DataFrame.fromCsvString("a\n1", { naValues: "NA" as unknown as string[] })
    ).toThrow(/naValues/);
  });

  it("accepts True/TRUE/False/FALSE as booleans", () => {
    const df = DataFrame.fromCsvString("a,b,c\nTrue,FALSE,true\nTRUE,False,false\n");
    expect(col(df, "a")).toEqual([true, true]);
    expect(col(df, "b")).toEqual([false, false]);
    expect(col(df, "c")).toEqual([true, false]);
  });

  it("dedupes duplicate headers like pandas", () => {
    // pandas: columns a, a.1, b
    const df = DataFrame.fromCsvString("a,a,b\n1,2,3\n");
    expect(df.columns).toEqual(["a", "a.1", "b"]);
    expect(col(df, "a.1")).toEqual([2]);
  });

  it("pads short rows with null and still rejects rows longer than the header", () => {
    // pandas: c = [nan, 5.0]
    const df = DataFrame.fromCsvString("a,b,c\n1,2\n3,4,5\n");
    expect(col(df, "c")).toEqual([null, 5]);
    expect(() => DataFrame.fromCsvString("a,b\n1,2,3\n")).toThrow(/Row 2 has 3 fields, expected 2/);
  });
});
