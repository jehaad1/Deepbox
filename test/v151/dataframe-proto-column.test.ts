import { describe, expect, it } from "vitest";
import { DataFrame } from "../../src/dataframe/index";

const json = '{"columns":["__proto__","b"],"index":[0,1],"data":{"__proto__":[5,6],"b":[1,2]}}';

describe("filter and assign(fn) can read a __proto__ column (DF-9)", () => {
  it("filter passes rows whose __proto__ field holds the column value", () => {
    const df = DataFrame.fromJsonString(json);
    const seen: unknown[] = [];
    const out = df.filter((row) => {
      seen.push(Object.getOwnPropertyDescriptor(row, "__proto__")?.value);
      return Object.getOwnPropertyDescriptor(row, "__proto__")?.value === 6;
    });
    expect(seen).toEqual([5, 6]);
    expect(out.shape).toEqual([1, 2]);
    expect(out.get("b").toArray()).toEqual([2]);
  });

  it("assign(fn) passes rows whose __proto__ field holds the column value", () => {
    const df = DataFrame.fromJsonString(json);
    const out = df.assign({
      c: (row) => (Object.getOwnPropertyDescriptor(row, "__proto__")?.value as number) * 10,
    });
    expect(out.get("c").toArray()).toEqual([50, 60]);
  });

  it("an object value in a __proto__ column does not replace the row prototype", () => {
    const df = new DataFrame({ a: [1] });
    const out = df.assign({ ["__proto__"]: [{ polluted: true }] });
    let polluted: unknown;
    out.filter((row) => {
      polluted = (row as { polluted?: boolean }).polluted;
      return true;
    });
    expect(polluted).toBeUndefined();
  });
});
