import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { DataValidationError, InvalidParameterError } from "../../src/core/errors";
import {
  DataFrame,
  date_range,
  MultiIndex,
  Series,
  timedelta,
  to_datetime,
} from "../../src/dataframe";

/**
 * Regression tests for the v1.5.0 review of Series, the string/datetime
 * accessors, MultiIndex, PlotAccessor and StyleAccessor.
 * Reference values come from pandas 3.0 / numpy 2.4 / Python's math.fsum.
 */

// ─── Series ──────────────────────────────────────────────────────────────

describe("Series (v1.5.0)", () => {
  it("tail(n) with n larger than the length returns every element", () => {
    const s = new Series([1, 2, 3], { index: ["a", "b", "c"] });
    const t = s.tail(5);
    expect(t.data).toEqual([1, 2, 3]);
    expect(t.index).toEqual(["a", "b", "c"]);
    expect(s.tail().data).toEqual([1, 2, 3]);
    expect(s.tail(0).data).toEqual([]);
    expect(s.tail(2).data).toEqual([2, 3]);
  });

  it("iloc rejects non-integer positions", () => {
    const s = new Series([10, 20, 30]);
    expect(() => s.iloc(1.5)).toThrow(InvalidParameterError);
    expect(() => s.iloc(Number.NaN)).toThrow(InvalidParameterError);
    expect(s.iloc(2)).toBe(30);
  });

  it("sort puts null, undefined and NaN last in both directions", () => {
    const s = new Series<number | null | undefined>([3, null, 1, Number.NaN, 2, undefined]);
    expect(s.sort().data.slice(0, 3)).toEqual([1, 2, 3]);
    expect(s.sort(false).data.slice(0, 3)).toEqual([3, 2, 1]);
    const asc = s.sort().data.slice(3);
    expect(asc.filter((v) => v === null || v === undefined || Number.isNaN(v))).toHaveLength(3);
    const strs = new Series<string | null>(["b", null, "a"]);
    expect(strs.sort().data).toEqual(["a", "b", null]);
    expect(strs.sort(false).data).toEqual(["b", "a", null]);
  });

  it("sort orders Dates chronologically and Infinity correctly", () => {
    const d = [new Date(2024, 5, 1), new Date(2023, 0, 1), new Date(2025, 0, 1)];
    const sorted = new Series(d).sort();
    expect(sorted.data.map((x) => x.getFullYear())).toEqual([2023, 2024, 2025]);
    const inf = new Series([Infinity, 1, -Infinity, Infinity]);
    expect(inf.sort().data).toEqual([-Infinity, 1, Infinity, Infinity]);
  });

  it("sort is stable and keeps the index aligned", () => {
    const s = new Series([2, 1, 2, 1], { index: ["a", "b", "c", "d"] });
    const r = s.sort();
    expect(r.data).toEqual([1, 1, 2, 2]);
    expect(r.index).toEqual(["b", "d", "a", "c"]);
  });

  it("valueCounts drops missing values by default (pandas dropna=True)", () => {
    const s = new Series<string | null>(["a", null, "a", "b"]);
    const vc = s.valueCounts();
    expect([...vc.index]).toEqual(["a", "b"]);
    expect(vc.data).toEqual([2, 1]);
    expect(new Series([1, Number.NaN, 1]).valueCounts().data).toEqual([2]);
  });

  it("valueCounts(false) keeps missing values without clashing with a real 'null' label", () => {
    const s = new Series<string | null>(["null", null, null, "x"]);
    const vc = s.valueCounts(false);
    expect(vc.length).toBe(3);
    expect(new Set(vc.index).size).toBe(3);
    expect(vc.loc("null")).toBe(1);
    expect(vc.data).toEqual([2, 1, 1]);
  });

  it("valueCounts keeps first-appearance order for ties", () => {
    const vc = new Series(["x", "y", "z", "y"]).valueCounts();
    expect([...vc.index]).toEqual(["y", "x", "z"]);
  });

  it("sum uses compensated summation (reference: math.fsum)", () => {
    expect(new Series(Array.from({ length: 10 }, () => 0.1)).sum()).toBe(1);
    expect(new Series([1e16, 1, -1e16]).sum()).toBe(1);
    const x = Array.from({ length: 1000 }, (_, i) => 0.1 * (i + 1));
    expect(new Series(x).sum()).toBeCloseTo(50050, 10);
  });

  it("sum and mean keep infinities and NaN semantics", () => {
    expect(new Series([1, Infinity]).sum()).toBe(Infinity);
    expect(new Series([Infinity, -Infinity]).sum()).toBeNaN();
    expect(new Series([1, Infinity]).mean()).toBe(Infinity);
    expect(new Series([1, 2, null, Number.NaN]).sum()).toBe(3);
    expect(new Series([null, null]).sum()).toBe(0);
    expect(new Series([null, null]).mean()).toBeNaN();
  });

  it("var/std stay accurate with a large offset and honor ddof (reference: numpy)", () => {
    const s = new Series([1e9 + 4, 1e9 + 7, 1e9 + 13, 1e9 + 16]);
    expect(s.var()).toBe(30);
    expect(s.std()).toBeCloseTo(5.477225575051661, 14);
    expect(s.var(0)).toBe(22.5);
    expect(s.std(0)).toBeCloseTo(4.743416490252569, 14);
    const t = new Series([2, 4, 6, 8]);
    expect(t.var()).toBeCloseTo(6.666666666666667, 14);
    expect(t.var(2)).toBeCloseTo(10, 14);
  });

  it("var/std return NaN when there are not enough values and never go negative", () => {
    expect(new Series([5]).var()).toBeNaN();
    expect(new Series([5]).var(0)).toBe(0);
    expect(new Series([0.1, 0.1, 0.1]).var()).toBe(0);
    expect(new Series([0.1, 0.1, 0.1]).std()).toBe(0);
  });

  it("var/std validate ddof", () => {
    const s = new Series([1, 2, 3]);
    expect(() => s.var(-1)).toThrow(InvalidParameterError);
    expect(() => s.std(1.5)).toThrow(InvalidParameterError);
  });

  it("median does not overflow for huge middle values", () => {
    expect(new Series([1.7e308, 1.7e308]).median()).toBe(1.7e308);
    expect(new Series([1, 2, 3, 4]).median()).toBe(2.5);
    expect(new Series([3, 1, 2]).median()).toBe(2);
    expect(new Series([Number.NaN, null]).median()).toBeNaN();
  });

  it("min/max skip missing values and report NaN when none remain", () => {
    expect(new Series([3, null, 1, Number.NaN]).min()).toBe(1);
    expect(new Series([3, null, 1, Number.NaN]).max()).toBe(3);
    expect(new Series([null]).min()).toBeNaN();
    expect(new Series([Infinity]).min()).toBe(Infinity);
    expect(() => new Series(["a"]).min()).toThrow(DataValidationError);
  });

  it("toString validates maxRows", () => {
    const s = new Series([1, 2, 3]);
    expect(() => s.toString(-1)).toThrow(InvalidParameterError);
    expect(() => s.toString(1.5)).toThrow(InvalidParameterError);
    expect(s.toString(2)).toContain("...");
    expect(s.toString(Number.POSITIVE_INFINITY)).not.toContain("...");
  });

  it("sort puts invalid Dates last and median handles overflow and subnormals", () => {
    const d = new Series([new Date("x"), new Date(2024, 0, 1), new Date(2023, 0, 1)]);
    const sorted = d.sort().data as Date[];
    expect(sorted[0]?.getFullYear()).toBe(2023);
    expect(Number.isNaN(sorted[2]?.getTime())).toBe(true);
    const huge = new Series([1.7e308, 1.6e308]).median();
    expect(Math.abs(huge / 1.65e308 - 1)).toBeLessThan(1e-15);
    expect(new Series([5e-324, 5e-324 * 3]).median()).toBe(1e-323);
  });
});

// ─── DateTimeAccessor ────────────────────────────────────────────────────

const originalTZ = process.env["TZ"];

function setTZ(tz: string | undefined): void {
  if (tz === undefined) {
    delete process.env["TZ"];
  } else {
    process.env["TZ"] = tz;
  }
}

describe("dt accessor (v1.5.0)", () => {
  it("dayofyear and weekofyear match Python's isocalendar around year boundaries", () => {
    // reference: pandas dt.dayofyear / dt.isocalendar().week
    const s = new Series([
      new Date(2024, 2, 31),
      new Date(2024, 2, 10),
      new Date(2021, 0, 3),
      new Date(2020, 11, 31),
      new Date(2024, 11, 30),
      new Date(2026, 0, 1),
    ]);
    expect(s.dt.dayofyear().data).toEqual([91, 70, 3, 366, 365, 1]);
    expect(s.dt.weekofyear().data).toEqual([13, 10, 53, 53, 1, 1]);
    expect(s.dt.week().data).toEqual([13, 10, 53, 53, 1, 1]);
  });

  describe("across a DST change", () => {
    beforeAll(() => setTZ("America/New_York"));
    afterAll(() => setTZ(originalTZ));

    it("dayofyear is exact for local midnights after spring-forward", () => {
      // 2024-03-10 is the US spring-forward day: midnight to midnight is 23 hours.
      const s = new Series([new Date(2024, 2, 31), new Date(2024, 2, 11), new Date(2024, 10, 3)]);
      expect(s.dt.dayofyear().data).toEqual([91, 71, 308]);
    });

    it("date_range('D') keeps midnight across the DST change", () => {
      const r = date_range(new Date(2024, 2, 8), 6, "D");
      expect(r.dt.hour().data).toEqual([0, 0, 0, 0, 0, 0]);
      expect(r.dt.day().data).toEqual([8, 9, 10, 11, 12, 13]);
    });

    it("ceil and round to 'D' land on the next local midnight", () => {
      const s = new Series([new Date(2024, 2, 10, 12, 0, 0)]);
      const c = s.dt.ceil("D").data[0] as Date;
      expect([c.getDate(), c.getHours()]).toEqual([11, 0]);
      const f = s.dt.floor("D").data[0] as Date;
      expect([f.getDate(), f.getHours()]).toEqual([10, 0]);
    });
  });

  it("date_range('D') never drifts off midnight (any time zone)", () => {
    const r = date_range("2024-01-01", 400, "D");
    expect(r.dt.hour().data.every((h) => h === 0)).toBe(true);
    expect(r.dt.dayofyear().data.slice(0, 3)).toEqual([1, 2, 3]);
  });

  it("round breaks exact ties to even, like pandas", () => {
    // reference: pandas dt.round("min") / dt.round("D") / dt.round("s")
    const t = new Series([
      new Date(2024, 0, 1, 0, 0, 30),
      new Date(2024, 0, 1, 0, 1, 30),
      new Date(2024, 0, 1, 12, 0, 0),
      new Date(2024, 0, 2, 12, 0, 0),
    ]);
    const min = t.dt.round("min").data as Date[];
    expect(min.map((d) => d.getMinutes())).toEqual([0, 2, 0, 0]);
    const day = t.dt.round("D").data as Date[];
    expect(day.map((d) => d.getDate())).toEqual([1, 1, 2, 2]);
    const sec = new Series([
      new Date(2024, 0, 1, 0, 0, 0, 500),
      new Date(2024, 0, 1, 0, 0, 1, 500),
    ]).dt.round("s").data as Date[];
    expect(sec.map((d) => d.getSeconds())).toEqual([0, 2]);
  });

  it("floor, ceil and round accept pandas' lowercase 'h' and 's'", () => {
    const s = new Series([new Date(2024, 0, 1, 10, 40, 20)]);
    expect((s.dt.floor("h").data[0] as Date).getMinutes()).toBe(0);
    expect((s.dt.ceil("h").data[0] as Date).getHours()).toBe(11);
    expect((s.dt.round("s").data[0] as Date).getSeconds()).toBe(20);
  });

  it("floor/ceil/round reject unknown frequencies", () => {
    const s = new Series([new Date(2024, 0, 1)]);
    // @ts-expect-error invalid frequency on purpose
    expect(() => s.dt.floor("W")).toThrow(InvalidParameterError);
    // @ts-expect-error invalid frequency on purpose
    expect(() => s.dt.ceil("x")).toThrow(InvalidParameterError);
    // @ts-expect-error invalid frequency on purpose
    expect(() => s.dt.round("")).toThrow(InvalidParameterError);
  });

  it("date_range MS and YS follow pandas anchoring", () => {
    // reference: pd.date_range("2024-01-15", periods=3, freq="MS") -> Feb 1, Mar 1, Apr 1
    const ms = date_range(new Date(2024, 0, 15), 3, "MS");
    expect(ms.dt.month().data).toEqual([2, 3, 4]);
    expect(ms.dt.day().data).toEqual([1, 1, 1]);
    // start on the 1st is kept, time of day is kept
    const ms2 = date_range(new Date(2024, 0, 1, 10, 0), 3, "MS");
    expect(ms2.dt.month().data).toEqual([1, 2, 3]);
    expect(ms2.dt.hour().data).toEqual([10, 10, 10]);
    // reference: pd.date_range("2024-01-31", periods=3, freq="MS") -> Feb 1, Mar 1, Apr 1
    expect(date_range(new Date(2024, 0, 31), 3, "MS").dt.month().data).toEqual([2, 3, 4]);
    // reference: pd.date_range("2024-03-15", periods=3, freq="YS") -> 2025, 2026, 2027
    const ys = date_range(new Date(2024, 2, 15), 3, "YS");
    expect(ys.dt.year().data).toEqual([2025, 2026, 2027]);
    expect(ys.dt.month().data).toEqual([1, 1, 1]);
    expect(date_range(new Date(2024, 0, 1), 2, "YS").dt.year().data).toEqual([2024, 2025]);
  });

  it("date_range validates freq and accepts Series options", () => {
    // @ts-expect-error invalid frequency on purpose
    expect(() => date_range("2024-01-01", 3, "W")).toThrow(InvalidParameterError);
    const r = date_range("2024-01-01", 2, "D", { name: "when", index: ["a", "b"] });
    expect(r.name).toBe("when");
    expect(r.index).toEqual(["a", "b"]);
    expect(date_range(new Date(2024, 0, 1, 0, 0, 0), 3, "h").dt.hour().data).toEqual([0, 1, 2]);
  });

  it("strftime supports the common directives (reference: pandas)", () => {
    const s = new Series([new Date(2024, 0, 15, 13, 5, 9, 123)]);
    // pandas: '24 01 PM Mon Monday Jan January 015 1 1 % 2024-01-15 13:05:09'
    expect(s.dt.strftime("%y %m %p %a %A %b %B %j %w %u %% %F %T").data[0]).toBe(
      "24 01 PM Mon Monday Jan January 015 1 1 % 2024-01-15 13:05:09"
    );
    expect(s.dt.strftime("%Y-%m-%d %H:%M:%S.%f").data[0]).toBe("2024-01-15 13:05:09.123000");
    expect(s.dt.strftime("%I:%M %p").data[0]).toBe("01:05 PM");
    const midnight = new Series([new Date(2024, 0, 1, 0, 30), new Date(2024, 0, 1, 12, 30)]);
    expect(midnight.dt.strftime("%I %p").data).toEqual(["12 AM", "12 PM"]);
    // unknown directives pass through, "%%Y" is a literal percent followed by Y
    expect(s.dt.strftime("%Q").data[0]).toBe("%Q");
    expect(s.dt.strftime("%%Y").data[0]).toBe("%Y");
  });

  it("day_name, month_name and is_*_start/end match pandas", () => {
    const s = new Series([new Date(2024, 0, 15)]);
    expect(s.dt.day_name().data).toEqual(["Monday"]);
    expect(s.dt.month_name().data).toEqual(["January"]);
    expect(s.dt.weekday().data).toEqual([0]);
    const e = new Series([
      new Date(2024, 1, 29),
      new Date(2024, 2, 31),
      new Date(2024, 0, 1),
      new Date(2024, 11, 31),
      new Date(2024, 5, 30),
    ]);
    expect(e.dt.is_month_end().data).toEqual([true, true, false, true, true]);
    const q = new Series([
      new Date(2024, 0, 1),
      new Date(2024, 2, 31),
      new Date(2024, 3, 1),
      new Date(2024, 11, 31),
      new Date(2024, 5, 30),
      new Date(2024, 5, 29),
    ]);
    expect(q.dt.is_quarter_start().data).toEqual([true, false, true, false, false, false]);
    expect(q.dt.is_quarter_end().data).toEqual([false, true, false, true, true, false]);
    expect(q.dt.is_year_start().data).toEqual([true, false, false, false, false, false]);
    expect(q.dt.is_year_end().data).toEqual([false, false, false, true, false, false]);
    expect(e.dt.is_month_start().data).toEqual([false, false, true, false, false]);
    expect(e.dt.daysinmonth().data).toEqual([29, 31, 31, 31, 30]);
  });

  it("normalize() truncates to midnight", () => {
    const s = new Series([new Date(2024, 0, 15, 10, 30)]);
    const d = s.dt.normalize().data[0] as Date;
    expect([d.getDate(), d.getHours(), d.getMinutes()]).toEqual([15, 0, 0]);
  });

  it("to_datetime rejects out-of-range components instead of rolling them over", () => {
    expect(() => to_datetime(["2024-02-30"])).toThrow(DataValidationError);
    expect(() => to_datetime(["2024-13-01"])).toThrow(DataValidationError);
    expect(() => to_datetime(["2024-00-10"])).toThrow(DataValidationError);
    expect(() => to_datetime(["2024-01-01 24:00"])).toThrow(DataValidationError);
    expect(() => to_datetime(["2024-01-01T10:61:00"])).toThrow(DataValidationError);
    expect(() => to_datetime(["2023-02-29"])).toThrow(DataValidationError);
    const ok = to_datetime(["2024-02-29", "2024-12-31T23:59:59"]);
    expect(ok.dt.day().data).toEqual([29, 31]);
  });

  it("to_datetime treats NaN and empty strings as missing", () => {
    const r = to_datetime(["2024-01-01", "", "  ", Number.NaN, null, undefined]);
    expect(r.data.map((d) => d === null)).toEqual([false, true, true, true, true, true]);
    expect(() => to_datetime([Infinity])).toThrow(DataValidationError);
  });

  it("to_datetime truncates fractional seconds to milliseconds", () => {
    const r = to_datetime([
      "2024-01-01 10:00:00.9996",
      "2024-01-01 10:00:00.5",
      "2024-01-01T10:00",
    ]);
    expect(r.dt.millisecond().data).toEqual([999, 500, 0]);
    expect(r.dt.second().data).toEqual([0, 0, 0]);
  });

  it("to_datetime returns new Date objects and rejects out-of-range epochs", () => {
    const src = new Date(2024, 0, 1);
    const r = to_datetime([src]);
    expect(r.data[0]).not.toBe(src);
    expect(r.data[0]?.getTime()).toBe(src.getTime());
    src.setFullYear(1999);
    expect(r.data[0]?.getFullYear()).toBe(2024);
    expect(() => to_datetime([1e20])).toThrow(DataValidationError);
  });

  it("parses years below 100 literally", () => {
    const r = to_datetime(["0050-06-15"]);
    expect(r.dt.year().data).toEqual([50]);
  });

  it("timedelta rejects unknown units", () => {
    const a = new Series([new Date(2024, 0, 2)]);
    const b = new Series([new Date(2024, 0, 1)]);
    // @ts-expect-error invalid unit on purpose
    expect(() => timedelta(a, b, "W")).toThrow(InvalidParameterError);
    expect(timedelta(a, b, "H").data).toEqual([24]);
  });
});

// ─── StringAccessor ──────────────────────────────────────────────────────

describe("str accessor (v1.5.0)", () => {
  it("title matches Python/pandas (lower-cases the rest, handles non-ASCII)", () => {
    // reference: pandas str.title()
    const s = new Series(["hELLO wORLD's 3rd", "élan vital", "  a  b ", "ab", "x,y;z"]);
    expect(s.str.title().data).toEqual([
      "Hello World'S 3Rd",
      "Élan Vital",
      "  A  B ",
      "Ab",
      "X,Y;Z",
    ]);
  });

  it("capitalize keeps astral first characters intact", () => {
    expect(new Series(["😀abc", "éCOLE"]).str.capitalize().data).toEqual(["😀abc", "École"]);
  });

  it("swapcase matches pandas", () => {
    expect(new Series(["Hello", "HELLO", "héLLO"]).str.swapcase().data).toEqual([
      "hELLO",
      "hello",
      "HÉllo",
    ]);
  });

  it("match anchors at the start while contains searches anywhere (reference: pandas)", () => {
    const s = new Series(["abc", "xabc", null]);
    expect(s.str.match("b").data).toEqual([false, false, null]);
    expect(s.str.match("a").data).toEqual([true, false, null]);
    expect(s.str.match("a|x").data).toEqual([true, true, null]);
    expect(s.str.contains("b").data).toEqual([true, true, null]);
  });

  it("fullmatch needs the whole string to match, with backtracking", () => {
    // reference: pandas str.fullmatch("ab")
    expect(new Series(["abc", "ab"]).str.fullmatch("ab").data).toEqual([false, true]);
    expect(new Series(["ab", "a"]).str.fullmatch("a|ab").data).toEqual([true, true]);
  });

  it("global and sticky RegExps do not leak lastIndex between elements", () => {
    const s = new Series(["a", "a", "a"]);
    expect(s.str.contains(/a/g).data).toEqual([true, true, true]);
    expect(s.str.match(/a/g).data).toEqual([true, true, true]);
    expect(s.str.contains(/a/y).data).toEqual([true, true, true]);
    expect(s.str.extract(/(a)/g, 1).data).toEqual(["a", "a", "a"]);
    expect(s.str.fullmatch(/a/g).data).toEqual([true, true, true]);
  });

  it("contains can ignore case", () => {
    // reference: pandas str.contains("b", case=False)
    expect(new Series(["ABC", "abc", "xyz"]).str.contains("b", true, false).data).toEqual([
      true,
      true,
      false,
    ]);
    expect(new Series(["ABC"]).str.contains("a.c", false, false).data).toEqual([false]);
  });

  it("default split drops leading/trailing whitespace like pandas", () => {
    // reference: pandas str.split()
    const s = new Series(["  a  b ", "hELLO wORLD's 3rd", ""]);
    expect(s.str.split().data).toEqual([["a", "b"], ["hELLO", "wORLD's", "3rd"], []]);
  });

  it("split(pat, n) keeps the unsplit remainder verbatim", () => {
    // reference: pandas str.split(",", n=2)
    expect(new Series(["a,b,c,d"]).str.split(",", 2).data).toEqual([["a", "b", "c,d"]]);
    // reference: pandas str.split(r"\d+", n=2, regex=True) -> ['a','b','c333d']
    expect(new Series(["a1b22c333d"]).str.split(/\d+/, 2).data).toEqual([["a", "b", "c333d"]]);
    // n = 0 and n = -1 both mean "all splits" in pandas
    expect(new Series(["a b", "c"]).str.split(" ", 0).data).toEqual([["a", "b"], ["c"]]);
    expect(new Series(["a b", "c"]).str.split(" ", -1).data).toEqual([["a", "b"], ["c"]]);
    // whitespace default with a limit keeps the remainder
    expect(new Series(["a  b  c "]).str.split(undefined, 1).data).toEqual([["a", "b  c "]]);
  });

  it("split keeps captured text like Python's re.split and ignores empty matches", () => {
    // reference: re.split(r"(\d)", "a1b2c") -> ['a', '1', 'b', '2', 'c']
    expect(new Series(["a1b2c"]).str.split(/(\d)/).data).toEqual([["a", "1", "b", "2", "c"]]);
    // reference: re.split(r"(\d)", "a1b2c", maxsplit=1) -> ['a', '1', 'b2c']
    expect(new Series(["a1b2c"]).str.split(/(\d)/, 1).data).toEqual([["a", "1", "b2c"]]);
    expect(new Series(["a b"]).str.split(/\s*/).data).toEqual([["a", "b"]]);
  });

  it("split with an empty separator gives characters; a fractional n is rejected", () => {
    expect(new Series(["a😀c"]).str.split("").data).toEqual([["a", "😀", "c"]]);
    expect(new Series(["abcd"]).str.split("", 2).data).toEqual([["a", "b", "cd"]]);
    expect(() => new Series(["abc"]).str.split(",", 1.5)).toThrow(InvalidParameterError);
  });

  it("split on whitespace uses Python's whitespace set", () => {
    expect(new Series(["a\u00a0b\u2003c\u0085d\ufeffe"]).str.split().data).toEqual([
      ["a", "b", "c", "d\ufeffe"],
    ]);
  });

  it("center/pad(both) put the odd fill character where Python does", () => {
    // reference: pandas str.center(5, "-") -> ['--ab-', '-abc-', '-abcd']
    expect(new Series(["ab", "abc", "abcd"]).str.center(5, "-").data).toEqual([
      "--ab-",
      "-abc-",
      "-abcd",
    ]);
    expect(new Series(["ab"]).str.center(6, "-").data).toEqual(["--ab--"]);
    expect(new Series(["ab"]).str.ljust(5, "*").data).toEqual(["ab***"]);
    expect(new Series(["ab"]).str.rjust(5, "*").data).toEqual(["***ab"]);
  });

  it("pad validates side and fillchar", () => {
    const s = new Series(["a"]);
    // @ts-expect-error invalid side on purpose
    expect(() => s.str.pad(3, "middle")).toThrow(InvalidParameterError);
    expect(() => s.str.pad(3, "left", "ab")).toThrow(InvalidParameterError);
    expect(() => s.str.pad(-1)).toThrow(InvalidParameterError);
    expect(s.str.pad(3, "left", "😀").data).toEqual(["😀😀a"]);
  });

  it("len, slice, pad and zfill count code points", () => {
    // reference: pandas str.len() -> 4, str.slice(2, 3) -> '😀', str.pad(6) -> two spaces
    const s = new Series(["ab😀c"]);
    expect(s.str.len().data).toEqual([4]);
    expect(s.str.slice(2, 3).data).toEqual(["😀"]);
    expect(s.str.pad(6).data).toEqual(["  ab😀c"]);
    expect(new Series(["😀"]).str.zfill(3).data).toEqual(["00😀"]);
  });

  it("slice supports a step with Python semantics", () => {
    // reference: pandas str.slice(0, None, 2) / slice(None, None, -1) / slice(4, 1, -2)
    expect(new Series(["abc"]).str.slice(0, undefined, 2).data).toEqual(["ac"]);
    expect(new Series(["abcdef"]).str.slice(undefined, undefined, -1).data).toEqual(["fedcba"]);
    expect(new Series(["abcdef"]).str.slice(4, 1, -2).data).toEqual(["ec"]);
    expect(() => new Series(["a"]).str.slice(0, 1, 0)).toThrow(InvalidParameterError);
    expect(() => new Series(["a"]).str.slice(0.5)).toThrow(InvalidParameterError);
  });

  it("get returns the character at a position or null", () => {
    // reference: pandas str.get(1) / get(-1) / get(5)
    const s = new Series(["abc", "xyz"]);
    expect(s.str.get(1).data).toEqual(["b", "y"]);
    expect(s.str.get(-1).data).toEqual(["c", "z"]);
    expect(s.str.get(5).data).toEqual([null, null]);
  });

  it("replace with regex=false inserts the replacement literally", () => {
    expect(new Series(["a.b"]).str.replace(".", "$&", false).data).toEqual(["a$&b"]);
    expect(new Series(["a.b"]).str.replace("a", "$1", false).data).toEqual(["$1.b"]);
    // Python: "abc".replace("", "-") -> "-a-b-c-"
    expect(new Series(["abc"]).str.replace("", "-", false).data).toEqual(["-a-b-c-"]);
    expect(new Series(["", "😀"]).str.replace("", "-", false).data).toEqual(["-", "-😀-"]);
    // regex mode still supports JS replacement syntax
    expect(new Series(["ab"]).str.replace("(a)", "[$1]").data).toEqual(["[a]b"]);
  });

  it("strip with characters treats regex metacharacters literally and is linear time", () => {
    expect(new Series(["^^a-]"]).str.strip("^-]").data).toEqual(["a"]);
    expect(new Series(["\\a\\"]).str.strip("\\").data).toEqual(["a"]);
    expect(new Series(["xxabcxx"]).str.lstrip("x").data).toEqual(["abcxx"]);
    expect(new Series(["xxabcxx"]).str.rstrip("x").data).toEqual(["xxabc"]);
    expect(new Series(["abc"]).str.strip("").data).toEqual(["abc"]);
    expect(new Series(["😀a😀"]).str.strip("😀").data).toEqual(["a"]);
    const long = `b${"a".repeat(200000)}b`;
    const start = Date.now();
    expect(new Series([long]).str.strip("a").data[0]).toBe(long);
    expect(Date.now() - start).toBeLessThan(2000);
  });

  it("extract validates the group number", () => {
    const s = new Series(["ab12"]);
    expect(s.str.extract(/([a-z]+)(\d+)/, 2).data).toEqual(["12"]);
    expect(s.str.extract(/([a-z]+)(\d+)/, 0).data).toEqual(["ab12"]);
    expect(() => s.str.extract(/([a-z]+)/, 2)).toThrow(InvalidParameterError);
    expect(() => s.str.extract(/a/, -1)).toThrow(InvalidParameterError);
    expect(() => s.str.extract(/a/, 0.5)).toThrow(InvalidParameterError);
  });

  it("character classes use Unicode rules like Python", () => {
    // reference: pandas isalpha / isdigit / isalnum / isnumeric / isdecimal
    expect(new Series(["é", "é1", "", "abc"]).str.isalpha().data).toEqual([
      true,
      false,
      false,
      true,
    ]);
    expect(new Series(["١٢", "²", "x", ""]).str.isdigit().data).toEqual([true, true, false, false]);
    expect(new Series(["é1", "²", "x_"]).str.isalnum().data).toEqual([true, true, false]);
    expect(new Series(["12", "1.5", "x"]).str.isnumeric().data).toEqual([true, false, false]);
    expect(new Series(["²", "12"]).str.isdecimal().data).toEqual([false, true]);
    expect(new Series(["  　", "a ", ""]).str.isspace().data).toEqual([true, false, false]);
  });

  it("find and rfind return character positions", () => {
    // reference: pandas str.find("c") -> 2, rfind("c") -> 5, find("z") -> -1
    const s = new Series(["abcabc"]);
    expect(s.str.find("c").data).toEqual([2]);
    expect(s.str.rfind("c").data).toEqual([5]);
    expect(s.str.find("z").data).toEqual([-1]);
    expect(new Series(["😀ab"]).str.find("b").data).toEqual([2]);
  });

  it("removeprefix and removesuffix", () => {
    // reference: pandas str.removeprefix / removesuffix
    expect(new Series(["prefix_x", "x"]).str.removeprefix("prefix_").data).toEqual(["x", "x"]);
    expect(new Series(["x.csv", "x"]).str.removesuffix(".csv").data).toEqual(["x", "x"]);
  });

  it("get_dummies keeps '__proto__' as a normal column and splits characters for an empty separator", () => {
    const df = new Series(["a|__proto__", "b"]).str.get_dummies("|");
    expect(df.columns).toEqual(["__proto__", "a", "b"]);
    expect(df.get("__proto__").data).toEqual([1, 0]);
    expect(df.get("a").data).toEqual([1, 0]);
    expect(df.get("b").data).toEqual([0, 1]);
    expect(new Series(["ab", "b"]).str.get_dummies("").columns).toEqual(["a", "b"]);
  });

  it("cat validates sep", () => {
    // @ts-expect-error invalid separator on purpose
    expect(() => new Series(["a"]).str.cat(1)).toThrow(InvalidParameterError);
  });

  it("replace validates its replacement argument", () => {
    // @ts-expect-error invalid replacement on purpose
    expect(() => new Series(["a"]).str.replace("a", 1)).toThrow(InvalidParameterError);
  });

  it("contains rejects a non-string, non-RegExp pattern with a typed error", () => {
    // @ts-expect-error invalid pattern on purpose
    expect(() => new Series(["a"]).str.contains(1)).toThrow(InvalidParameterError);
  });
});

// ─── MultiIndex ──────────────────────────────────────────────────────────

describe("MultiIndex (v1.5.0)", () => {
  it("keeps 1 and '1' apart in lookups", () => {
    const mi = MultiIndex.fromTuples(
      [
        ["a", 1],
        ["a", "1"],
      ],
      ["k", "v"]
    );
    expect(mi.getPosition(["a", 1])).toBe(0);
    expect(mi.getPosition(["a", "1"])).toBe(1);
    expect(mi.isUnique).toBe(true);
  });

  it("separator characters inside labels cannot make two tuples collide", () => {
    const mi = MultiIndex.fromTuples(
      [
        ["a\0b", "c"],
        ["a", "b\0c"],
      ],
      ["x", "y"]
    );
    expect(mi.getPosition(["a\0b", "c"])).toBe(0);
    expect(mi.getPosition(["a", "b\0c"])).toBe(1);
  });

  it("getPosition returns the first of duplicate tuples and isUnique reports them", () => {
    const mi = MultiIndex.fromTuples(
      [
        ["a", 1],
        ["b", 2],
        ["a", 1],
      ],
      ["x", "y"]
    );
    expect(mi.getPosition(["a", 1])).toBe(0);
    expect(mi.isUnique).toBe(false);
    expect(mi.getPosition(["a"])).toBe(-1);
    expect(mi.has(["b", 2])).toBe(true);
    expect(mi.has(["b", 3])).toBe(false);
  });

  it("select validates level names even on an empty index", () => {
    const empty = MultiIndex.fromTuples([], ["a", "b"]);
    expect(empty.length).toBe(0);
    expect(empty.nlevels).toBe(2);
    expect(() => empty.select({ nope: 1 })).toThrow(DataValidationError);
    expect(empty.select({ a: 1 })).toEqual([]);
  });

  it("select can match NaN labels", () => {
    const mi = MultiIndex.fromArrays([[Number.NaN, 1, Number.NaN]], ["x"]);
    expect(mi.select({ x: Number.NaN })).toEqual([0, 2]);
  });

  it("level arguments are validated and negative positions count from the end", () => {
    const mi = MultiIndex.fromArrays(
      [
        ["US", "UK"],
        [2020, 2021],
      ],
      ["country", "year"]
    );
    expect(() => mi.getLevel(Number.NaN)).toThrow(InvalidParameterError);
    expect(() => mi.getLevel(1.5)).toThrow(InvalidParameterError);
    expect(() => mi.getLevel(2)).toThrow(InvalidParameterError);
    expect(() => mi.getLevel("missing")).toThrow(InvalidParameterError);
    expect(mi.getLevel(-1)).toEqual([2020, 2021]);
    expect(() => mi.droplevel(Number.NaN)).toThrow(InvalidParameterError);
    expect(() => mi.swaplevel(0, Number.NaN)).toThrow(InvalidParameterError);
    expect(mi.droplevel(-1).names).toEqual(["country"]);
    expect(mi.swaplevel(0, -1).names).toEqual(["year", "country"]);
  });

  it("a duplicated level name must be addressed by position", () => {
    const mi = MultiIndex.fromArrays([[1], [2]], ["x", "x"]);
    expect(() => mi.getLevel("x")).toThrow(DataValidationError);
    expect(mi.getLevel(1)).toEqual([2]);
  });

  it("get rejects fractional positions", () => {
    const mi = MultiIndex.fromArrays([[1, 2]], ["x"]);
    expect(() => mi.get(0.5)).toThrow(InvalidParameterError);
    expect(mi.get(1)).toEqual([2]);
  });

  it("fromProduct works for very large products and in row-major order", () => {
    const big = MultiIndex.fromProduct(
      [Array.from({ length: 500 }, (_, i) => i), Array.from({ length: 400 }, (_, i) => `s${i}`)],
      ["a", "b"]
    );
    expect(big.length).toBe(200000);
    expect(big.get(0)).toEqual([0, "s0"]);
    expect(big.get(399)).toEqual([0, "s399"]);
    expect(big.get(400)).toEqual([1, "s0"]);
    expect(big.get(199999)).toEqual([499, "s399"]);
    const three = MultiIndex.fromProduct([
      ["a", "b"],
      [1, 2],
      ["x", "y"],
    ]);
    expect(three.toTuples()).toEqual([
      ["a", 1, "x"],
      ["a", 1, "y"],
      ["a", 2, "x"],
      ["a", 2, "y"],
      ["b", 1, "x"],
      ["b", 1, "y"],
      ["b", 2, "x"],
      ["b", 2, "y"],
    ]);
  });

  it("fromProduct with an empty level gives an empty MultiIndex", () => {
    const mi = MultiIndex.fromProduct([["a", "b"], []], ["x", "y"]);
    expect(mi.length).toBe(0);
    expect(mi.nlevels).toBe(2);
  });

  it("fromTuples reports a missing label instead of inventing 0", () => {
    // @ts-expect-error a tuple with an undefined label on purpose
    expect(() => MultiIndex.fromTuples([["a", undefined]])).toThrow(DataValidationError);
    expect(() => MultiIndex.fromTuples([])).toThrow(InvalidParameterError);
  });

  it("fromArrays rejects non-array levels", () => {
    // @ts-expect-error not an array on purpose
    expect(() => MultiIndex.fromArrays([["a"], "b"])).toThrow(InvalidParameterError);
  });

  it("equals compares names, shape and labels", () => {
    const a = MultiIndex.fromArrays(
      [
        ["a", "b"],
        [1, 2],
      ],
      ["x", "y"]
    );
    const b = MultiIndex.fromArrays(
      [
        ["a", "b"],
        [1, 2],
      ],
      ["x", "y"]
    );
    const c = MultiIndex.fromArrays(
      [
        ["a", "b"],
        [1, 3],
      ],
      ["x", "y"]
    );
    const d = MultiIndex.fromArrays(
      [
        ["a", "b"],
        [1, 2],
      ],
      ["x", "z"]
    );
    expect(a.equals(b)).toBe(true);
    expect(a.equals(c)).toBe(false);
    expect(a.equals(d)).toBe(false);
    expect(a.equals(a.droplevel(0))).toBe(false);
  });

  it("toFlatIndex validates the separator", () => {
    const mi = MultiIndex.fromArrays([["a"], [1]]);
    // @ts-expect-error invalid separator on purpose
    expect(() => mi.toFlatIndex(1)).toThrow(InvalidParameterError);
    expect(mi.toFlatIndex("|")).toEqual(["(a|1)"]);
  });
});

// ─── PlotAccessor ────────────────────────────────────────────────────────

describe("df.plot (v1.5.0)", () => {
  const df = new DataFrame({
    x: [1, 2, 3],
    y: [4, 5, 6],
    z: [7, 8, 9],
    label: ["a", "b", "c"],
  });

  it("rejects unknown columns with the list of available ones", () => {
    expect(() => df.plot.line({ x: "nope" })).toThrow(/available columns: x, y, z, label/);
    expect(() => df.plot.line({ y: "nope" })).toThrow(InvalidParameterError);
    expect(() => df.plot.bar({ x: "nope" })).toThrow(InvalidParameterError);
    expect(() => df.plot.hist({ column: "nope" })).toThrow(InvalidParameterError);
    expect(() => df.plot.scatter({ x: "x", y: "nope" })).toThrow(InvalidParameterError);
    expect(() => df.plot.pie({ y: "y", labels: "nope" })).toThrow(InvalidParameterError);
  });

  it("rejects non-numeric values instead of plotting NaN", () => {
    expect(() => df.plot.scatter({ x: "label", y: "y" })).toThrow(DataValidationError);
    expect(() => df.plot.hist({ column: "label" })).toThrow(DataValidationError);
    expect(() => df.plot.pie({ y: "label" })).toThrow(DataValidationError);
  });

  it("defaults to the numeric columns only", () => {
    expect(() => df.plot.hist()).not.toThrow();
    expect(() => df.plot.box()).not.toThrow();
    expect(() => df.plot.line()).not.toThrow();
    expect(() => df.plot.area()).not.toThrow();
    const svg = df.plot.line({ x: "x" }).renderSVG().svg;
    expect(svg).toContain(">y<");
    expect(svg).toContain(">z<");
  });

  it("draws string columns as bar categories with tick labels", () => {
    const svg = df.plot.bar({ x: "label", y: "y" }).renderSVG().svg;
    for (const c of ["a", "b", "c"]) expect(svg).toContain(`>${c}<`);
    const h = df.plot.barh({ y: "label", x: "y" }).renderSVG().svg;
    for (const c of ["a", "b", "c"]) expect(h).toContain(`>${c}<`);
  });

  it("draws several y columns as grouped bars with a legend", () => {
    const svg = df.plot.bar({ x: "x", y: ["y", "z"] }).renderSVG().svg;
    expect(svg).toContain(">y<");
    expect(svg).toContain(">z<");
  });

  it("treats null as missing rather than zero", () => {
    const withNull = new DataFrame({ v: [1, null, 3], w: [4, 5, 6] });
    expect(() => withNull.plot.line({ y: "v" }).renderSVG()).not.toThrow();
    // A column that only holds null/undefined/numbers is numeric.
    expect(() => withNull.plot.hist().renderSVG()).not.toThrow();
  });

  it("area labels the x axis like line does", () => {
    expect(df.plot.area({ x: "x", y: "y" }).renderSVG().svg).toContain(">x<");
  });

  it("validates figsize", () => {
    expect(() => df.plot.line({ figsize: [0, 100] })).toThrow(InvalidParameterError);
    expect(() => df.plot.line({ figsize: [100, Number.NaN] })).toThrow(InvalidParameterError);
    expect(df.plot.line({ figsize: [300, 200] }).renderSVG().svg).toContain('width="300"');
  });
});

// ─── StyleAccessor ───────────────────────────────────────────────────────

describe("df.style (v1.5.0)", () => {
  const styleOf = (html: string, text: string): string | undefined => {
    const m = new RegExp(`<td( style="([^"]*)")?>${text}</td>`).exec(html);
    return m?.[2];
  };

  it("highlight_max and highlight_min mark every tied cell", () => {
    const df = new DataFrame({ a: [1, 5, 5, 0, 0] });
    const html = df.style
      .highlight_max({ color: "red" })
      .highlight_min({ fontWeight: "bold" })
      .toHTML();
    const cells = [...html.matchAll(/<td( style="([^"]*)")?>(\d)<\/td>/g)].map((m) => [m[3], m[2]]);
    expect(cells).toEqual([
      ["1", undefined],
      ["5", "color:red"],
      ["5", "color:red"],
      ["0", "font-weight:bold"],
      ["0", "font-weight:bold"],
    ]);
  });

  it("null is not treated as 0 when finding the minimum", () => {
    const df = new DataFrame({ a: [null, 3, 7] });
    const html = df.style.highlight_min({ color: "red" }).toHTML();
    expect(styleOf(html, "3")).toBe("color:red");
    expect(styleOf(html, "")).toBeUndefined();
  });

  it("strings, booleans and Dates are not numeric for highlighting", () => {
    const df = new DataFrame({ a: ["10", "2", true, new Date(0)] });
    const html = df.style.highlight_max({ color: "red" }).toHTML();
    expect(html).not.toContain("color:red");
  });

  it("highlight_null flags null, undefined and NaN but not Infinity", () => {
    const df = new DataFrame({ a: [null, Number.NaN, Infinity, 1] });
    const html = df.style.highlight_null({ color: "red" }).toHTML();
    expect([...html.matchAll(/style="color:red"/g)]).toHaveLength(2);
    expect(html).toContain("<td>Infinity</td>");
  });

  it("background_gradient and bar skip null and fill the whole range", () => {
    const df = new DataFrame({ a: [null, 0, 10] });
    const html = df.style.background_gradient("#000000", "#ffffff").toHTML();
    expect(html).toContain("background:rgb(0,0,0)");
    expect(html).toContain("background:rgb(255,255,255)");
    expect(html.match(/background:/g)).toHaveLength(2);
    const bar = df.style.bar("green").toHTML();
    expect(bar).toContain("green 0.0%");
    expect(bar).toContain("green 100.0%");
  });

  it("background_gradient rejects colors that are not hex", () => {
    const df = new DataFrame({ a: [1, 2] });
    expect(() => df.style.background_gradient("red")).toThrow(InvalidParameterError);
    expect(() => df.style.background_gradient("#ffffff", "#12")).toThrow(InvalidParameterError);
    expect(() => df.style.background_gradient("#fff", "#4472C4")).not.toThrow();
    expect(() => df.style.bar("")).toThrow(InvalidParameterError);
  });

  it("format rejects unknown columns", () => {
    const df = new DataFrame({ a: [1] });
    expect(() => df.style.format("b", String)).toThrow(/available columns: a/);
  });

  it("shows Dates as ISO strings and escapes quotes", () => {
    const df = new DataFrame({ d: [new Date(Date.UTC(2024, 0, 1))], t: [`it's "x"`] });
    const html = df.style.toHTML();
    expect(html).toContain("<td>2024-01-01T00:00:00.000Z</td>");
    expect(html).toContain("it&#39;s &quot;x&quot;");
  });

  it("map is an alias of applymap and toANSI knows more colors", () => {
    const df = new DataFrame({ a: [1] });
    const ansi = df.style.map(() => ({ color: "magenta" })).toANSI();
    expect(ansi).toContain("\x1b[35m1\x1b[0m");
    expect(df.style.map(() => ({ color: "gray" })).toANSI()).toContain("\x1b[90m");
    expect(df.style.map(() => ({ color: "#123456" })).toANSI()).not.toContain("\x1b[");
  });
});
