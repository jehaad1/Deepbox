import { describe, expect, it } from "vitest";
import { Series } from "../src/dataframe";

describe("StringAccessor", () => {
  // ─── Case transforms ───────────────────────────────────────────

  describe("upper()", () => {
    it("converts strings to uppercase", () => {
      const s = new Series(["hello", "world", "FoO"]);
      const result = s.str.upper();
      expect(result.data).toEqual(["HELLO", "WORLD", "FOO"]);
    });

    it("propagates null", () => {
      const s = new Series(["abc", null, "def"]);
      const result = s.str.upper();
      expect(result.data).toEqual(["ABC", null, "DEF"]);
    });

    it("handles empty strings", () => {
      const s = new Series([""]);
      expect(s.str.upper().data).toEqual([""]);
    });
  });

  describe("lower()", () => {
    it("converts strings to lowercase", () => {
      const s = new Series(["HELLO", "World"]);
      expect(s.str.lower().data).toEqual(["hello", "world"]);
    });

    it("propagates null", () => {
      const s = new Series([null, "ABC"]);
      expect(s.str.lower().data).toEqual([null, "abc"]);
    });
  });

  describe("title()", () => {
    it("title-cases each word", () => {
      const s = new Series(["hello world", "foo bar baz"]);
      expect(s.str.title().data).toEqual(["Hello World", "Foo Bar Baz"]);
    });

    it("propagates null", () => {
      const s = new Series([null]);
      expect(s.str.title().data).toEqual([null]);
    });
  });

  describe("capitalize()", () => {
    it("capitalizes the first character", () => {
      const s = new Series(["hello", "WORLD", "fOO"]);
      expect(s.str.capitalize().data).toEqual(["Hello", "World", "Foo"]);
    });

    it("handles empty string", () => {
      const s = new Series([""]);
      expect(s.str.capitalize().data).toEqual([""]);
    });
  });

  // ─── Trimming ──────────────────────────────────────────────────

  describe("strip()", () => {
    it("strips whitespace by default", () => {
      const s = new Series(["  hello  ", "\tworld\n"]);
      expect(s.str.strip().data).toEqual(["hello", "world"]);
    });

    it("strips specified characters", () => {
      const s = new Series(["xxhelloxx", "xyworldxy"]);
      expect(s.str.strip("xy").data).toEqual(["hello", "world"]);
    });

    it("propagates null", () => {
      const s = new Series([null, "  abc  "]);
      expect(s.str.strip().data).toEqual([null, "abc"]);
    });
  });

  describe("lstrip()", () => {
    it("strips leading whitespace", () => {
      const s = new Series(["  hello  "]);
      expect(s.str.lstrip().data).toEqual(["hello  "]);
    });

    it("strips leading characters", () => {
      const s = new Series(["xxhello"]);
      expect(s.str.lstrip("x").data).toEqual(["hello"]);
    });
  });

  describe("rstrip()", () => {
    it("strips trailing whitespace", () => {
      const s = new Series(["  hello  "]);
      expect(s.str.rstrip().data).toEqual(["  hello"]);
    });

    it("strips trailing characters", () => {
      const s = new Series(["helloxx"]);
      expect(s.str.rstrip("x").data).toEqual(["hello"]);
    });
  });

  // ─── Search / Match ────────────────────────────────────────────

  describe("contains()", () => {
    it("checks substring presence with regex", () => {
      const s = new Series(["hello", "world", "help"]);
      expect(s.str.contains("hel").data).toEqual([true, false, true]);
    });

    it("supports regex patterns", () => {
      const s = new Series(["hello123", "world", "abc456"]);
      expect(s.str.contains("\\d+").data).toEqual([true, false, true]);
    });

    it("supports literal mode", () => {
      const s = new Series(["a.b", "ab", "a.b.c"]);
      expect(s.str.contains("a.b", false).data).toEqual([true, false, true]);
    });

    it("propagates null", () => {
      const s = new Series(["hello", null]);
      expect(s.str.contains("hel").data).toEqual([true, null]);
    });
  });

  describe("startswith()", () => {
    it("checks prefix", () => {
      const s = new Series(["hello", "world", "help"]);
      expect(s.str.startswith("hel").data).toEqual([true, false, true]);
    });

    it("propagates null", () => {
      const s = new Series([null, "abc"]);
      expect(s.str.startswith("a").data).toEqual([null, true]);
    });
  });

  describe("endswith()", () => {
    it("checks suffix", () => {
      const s = new Series(["hello", "world"]);
      expect(s.str.endswith("llo").data).toEqual([true, false]);
    });
  });

  describe("match()", () => {
    it("checks regex match", () => {
      const s = new Series(["abc123", "xyz"]);
      expect(s.str.match(/\d+/).data).toEqual([true, false]);
    });
  });

  // ─── Replace / Split ──────────────────────────────────────────

  describe("replace()", () => {
    it("replaces with regex by default", () => {
      const s = new Series(["hello world", "foo bar"]);
      expect(s.str.replace("o", "0").data).toEqual(["hell0 w0rld", "f00 bar"]);
    });

    it("replaces literally when regex=false", () => {
      const s = new Series(["a.b.c"]);
      expect(s.str.replace(".", "-", false).data).toEqual(["a-b-c"]);
    });

    it("propagates null", () => {
      const s = new Series([null, "abc"]);
      expect(s.str.replace("a", "x").data).toEqual([null, "xbc"]);
    });
  });

  describe("split()", () => {
    it("splits by whitespace by default", () => {
      const s = new Series(["hello world", "foo  bar"]);
      const result = s.str.split();
      expect(result.data).toEqual([
        ["hello", "world"],
        ["foo", "bar"],
      ]);
    });

    it("splits by custom pattern", () => {
      const s = new Series(["a,b,c"]);
      expect(s.str.split(",").data).toEqual([["a", "b", "c"]]);
    });

    it("limits splits with n parameter", () => {
      const s = new Series(["a,b,c,d"]);
      expect(s.str.split(",", 2).data).toEqual([["a", "b", "c,d"]]);
    });

    it("propagates null", () => {
      const s = new Series([null]);
      expect(s.str.split().data).toEqual([null]);
    });
  });

  // ─── Length / Slice ────────────────────────────────────────────

  describe("len()", () => {
    it("returns string lengths", () => {
      const s = new Series(["hello", "hi", ""]);
      expect(s.str.len().data).toEqual([5, 2, 0]);
    });

    it("propagates null", () => {
      const s = new Series([null, "abc"]);
      expect(s.str.len().data).toEqual([null, 3]);
    });
  });

  describe("slice()", () => {
    it("slices strings", () => {
      const s = new Series(["hello", "world"]);
      expect(s.str.slice(0, 3).data).toEqual(["hel", "wor"]);
    });

    it("slices from start", () => {
      const s = new Series(["hello"]);
      expect(s.str.slice(2).data).toEqual(["llo"]);
    });

    it("handles negative indices", () => {
      const s = new Series(["hello"]);
      expect(s.str.slice(-3).data).toEqual(["llo"]);
    });
  });

  // ─── Extract / FindAll ─────────────────────────────────────────

  describe("extract()", () => {
    it("extracts first match", () => {
      const s = new Series(["abc123", "xyz456", "no-digits"]);
      expect(s.str.extract(/(\d+)/, 1).data).toEqual(["123", "456", null]);
    });

    it("returns null when no match", () => {
      const s = new Series(["abc"]);
      expect(s.str.extract(/\d+/).data).toEqual([null]);
    });
  });

  describe("findall()", () => {
    it("finds all matches", () => {
      const s = new Series(["abc123def456", "no match"]);
      expect(s.str.findall(/\d+/).data).toEqual([["123", "456"], []]);
    });

    it("propagates null", () => {
      const s = new Series([null]);
      expect(s.str.findall(/\d+/).data).toEqual([null]);
    });
  });

  // ─── Padding ───────────────────────────────────────────────────

  describe("pad()", () => {
    it("pads left by default", () => {
      const s = new Series(["hi", "hello"]);
      expect(s.str.pad(5).data).toEqual(["   hi", "hello"]);
    });

    it("pads right", () => {
      const s = new Series(["hi"]);
      expect(s.str.pad(5, "right").data).toEqual(["hi   "]);
    });

    it("pads both sides", () => {
      const s = new Series(["hi"]);
      expect(s.str.pad(6, "both").data).toEqual(["  hi  "]);
    });

    it("uses custom fill character", () => {
      const s = new Series(["hi"]);
      expect(s.str.pad(5, "left", "*").data).toEqual(["***hi"]);
    });

    it("does not truncate longer strings", () => {
      const s = new Series(["hello"]);
      expect(s.str.pad(3).data).toEqual(["hello"]);
    });
  });

  describe("center()", () => {
    it("centers string", () => {
      const s = new Series(["hi"]);
      expect(s.str.center(6, "-").data).toEqual(["--hi--"]);
    });
  });

  describe("zfill()", () => {
    it("zero-fills strings", () => {
      const s = new Series(["42", "7", "100"]);
      expect(s.str.zfill(5).data).toEqual(["00042", "00007", "00100"]);
    });

    it("preserves leading sign", () => {
      const s = new Series(["-42", "+7"]);
      expect(s.str.zfill(5).data).toEqual(["-0042", "+0007"]);
    });
  });

  // ─── Concatenation ─────────────────────────────────────────────

  describe("cat()", () => {
    it("concatenates with default empty separator", () => {
      const s = new Series(["a", "b", "c"]);
      expect(s.str.cat()).toBe("abc");
    });

    it("concatenates with separator", () => {
      const s = new Series(["a", "b", "c"]);
      expect(s.str.cat(", ")).toBe("a, b, c");
    });

    it("skips null values", () => {
      const s = new Series(["a", null, "c"]);
      expect(s.str.cat("-")).toBe("a-c");
    });
  });

  // ─── get_dummies ───────────────────────────────────────────────

  describe("get_dummies()", () => {
    it("creates dummy variables", () => {
      const s = new Series(["a|b", "b|c", "a"]);
      const result = s.str.get_dummies("|");
      expect(result.columns).toEqual(["a", "b", "c"]);
      expect(result.shape).toEqual([3, 3]);
      // Row 0: a|b -> a=1, b=1, c=0
      expect(result.iloc(0)).toEqual({ a: 1, b: 1, c: 0 });
      // Row 1: b|c -> a=0, b=1, c=1
      expect(result.iloc(1)).toEqual({ a: 0, b: 1, c: 1 });
      // Row 2: a -> a=1, b=0, c=0
      expect(result.iloc(2)).toEqual({ a: 1, b: 0, c: 0 });
    });

    it("handles null values", () => {
      const s = new Series(["a", null, "b"]);
      const result = s.str.get_dummies("|");
      // null row should be all zeros
      expect(result.iloc(1)).toEqual({ a: 0, b: 0 });
    });
  });

  // ─── Repeat ────────────────────────────────────────────────────

  describe("repeat()", () => {
    it("repeats strings", () => {
      const s = new Series(["ab", "x"]);
      expect(s.str.repeat(3).data).toEqual(["ababab", "xxx"]);
    });

    it("propagates null", () => {
      const s = new Series([null]);
      expect(s.str.repeat(2).data).toEqual([null]);
    });
  });

  // ─── Count ─────────────────────────────────────────────────────

  describe("count()", () => {
    it("counts pattern occurrences", () => {
      const s = new Series(["aabaa", "xyz"]);
      expect(s.str.count("a").data).toEqual([4, 0]);
    });

    it("counts regex matches", () => {
      const s = new Series(["abc123def456"]);
      expect(s.str.count(/\d+/).data).toEqual([2]);
    });
  });

  // ─── Boolean checks ───────────────────────────────────────────

  describe("isalpha()", () => {
    it("checks alpha characters", () => {
      const s = new Series(["abc", "abc123", "", "ABC"]);
      expect(s.str.isalpha().data).toEqual([true, false, false, true]);
    });
  });

  describe("isdigit()", () => {
    it("checks digit characters", () => {
      const s = new Series(["123", "abc", "12.3", ""]);
      expect(s.str.isdigit().data).toEqual([true, false, false, false]);
    });
  });

  describe("isalnum()", () => {
    it("checks alphanumeric characters", () => {
      const s = new Series(["abc123", "abc", "123", "a-b", ""]);
      expect(s.str.isalnum().data).toEqual([true, true, true, false, false]);
    });
  });

  describe("isspace()", () => {
    it("checks whitespace", () => {
      const s = new Series(["  ", "\t\n", "a b", ""]);
      expect(s.str.isspace().data).toEqual([true, true, false, false]);
    });
  });

  describe("isupper()", () => {
    it("checks uppercase", () => {
      const s = new Series(["ABC", "abc", "Abc", "123"]);
      expect(s.str.isupper().data).toEqual([true, false, false, false]);
    });
  });

  describe("islower()", () => {
    it("checks lowercase", () => {
      const s = new Series(["abc", "ABC", "Abc", "123"]);
      expect(s.str.islower().data).toEqual([true, false, false, false]);
    });
  });

  // ─── Error handling ────────────────────────────────────────────

  describe("error handling", () => {
    it("throws for non-string data", () => {
      const s = new Series([1, 2, 3]);
      expect(() => s.str.upper()).toThrow("not a string");
    });

    it("preserves index and name", () => {
      const s = new Series(["hello", "world"], {
        index: ["a", "b"],
        name: "words",
      });
      const result = s.str.upper();
      expect(result.index).toEqual(["a", "b"]);
      expect(result.name).toBe("words");
    });
  });
});
