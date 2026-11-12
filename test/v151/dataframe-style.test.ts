import { describe, expect, it } from "vitest";
import { DataFrame } from "../../src/dataframe/index";

const ESC = String.fromCharCode(27);

describe("StyleAccessor.toANSI escapes control characters (DF-24)", () => {
  it("shows control characters as visible escapes instead of emitting them", () => {
    const df = new DataFrame({ a: [`${ESC}[31mred${ESC}[0m`, "line1\nline2", "tab\there", "ok"] });
    const out = df.style.toANSI();
    expect(out).not.toContain(ESC);
    const rows = out.split("\n");
    for (const ch of rows.join("")) {
      const code = ch.charCodeAt(0);
      expect(code >= 32 && code !== 127 && !(code >= 128 && code <= 159)).toBe(true);
    }
    expect(out.split("\n").length).toBe(2 + 4);
    expect(out).toContain("\\x1b[31mred\\x1b[0m");
    expect(out).toContain("line1\\nline2");
    expect(out).toContain("tab\\there");
  });

  it("escapes control characters in column names too", () => {
    const df = new DataFrame({ [`a${ESC}b`]: [1] });
    expect(df.style.toANSI()).not.toContain(ESC);
  });

  it("keeps the ANSI styling it adds itself", () => {
    const df = new DataFrame({ a: [1] });
    expect(df.style.applymap(() => ({ fontWeight: "bold" })).toANSI()).toContain(`${ESC}[1m`);
  });
});

describe("StyleAccessor.toANSI aligns East Asian wide characters (DF-24)", () => {
  it("counts a wide character as two columns", () => {
    const df = new DataFrame({ name: ["日本語", "abc", "ab"], n: [1, 22, 333] });
    const lines = df.style.toANSI().split("\n");
    // "日本語" is 6 columns wide, so "abc" and "ab" are padded to 6 columns, not 3.
    expect(lines[0]).toBe("name    n  ");
    expect(lines[1]).toBe("------  ---");
    expect(lines[2]).toBe("日本語  1  ");
    expect(lines[3]).toBe("abc     22 ");
    expect(lines[4]).toBe("ab      333");
  });

  it("pads narrow columns by their text length", () => {
    const df = new DataFrame({ a: ["x", "yy"] });
    expect(df.style.toANSI().split("\n")).toEqual(["a ", "--", "x ", "yy"]);
  });
});
