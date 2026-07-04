import { describe, expect, it } from "vitest";
import { date_range, Series, timedelta, to_datetime } from "../src/dataframe";

describe("DateTimeAccessor", () => {
  const dates = [
    new Date(2024, 0, 15, 10, 30, 45), // Jan 15 2024 10:30:45
    new Date(2024, 5, 20, 14, 0, 0), // Jun 20 2024 14:00:00
    new Date(2023, 11, 31, 23, 59, 59), // Dec 31 2023 23:59:59
  ];

  describe("component extraction", () => {
    it("extracts year", () => {
      const s = new Series(dates);
      expect(s.dt.year().data).toEqual([2024, 2024, 2023]);
    });

    it("extracts month (1-indexed)", () => {
      const s = new Series(dates);
      expect(s.dt.month().data).toEqual([1, 6, 12]);
    });

    it("extracts day", () => {
      const s = new Series(dates);
      expect(s.dt.day().data).toEqual([15, 20, 31]);
    });

    it("extracts hour", () => {
      const s = new Series(dates);
      expect(s.dt.hour().data).toEqual([10, 14, 23]);
    });

    it("extracts minute", () => {
      const s = new Series(dates);
      expect(s.dt.minute().data).toEqual([30, 0, 59]);
    });

    it("extracts second", () => {
      const s = new Series(dates);
      expect(s.dt.second().data).toEqual([45, 0, 59]);
    });

    it("extracts millisecond", () => {
      const s = new Series([new Date(2024, 0, 1, 0, 0, 0, 123)]);
      expect(s.dt.millisecond().data).toEqual([123]);
    });

    it("extracts dayofweek (Mon=0 ... Sun=6)", () => {
      const s = new Series([
        new Date(2024, 0, 15), // Monday
        new Date(2024, 0, 21), // Sunday
        new Date(2024, 0, 17), // Wednesday
      ]);
      expect(s.dt.dayofweek().data).toEqual([0, 6, 2]);
    });

    it("extracts dayofyear", () => {
      const s = new Series([
        new Date(2024, 0, 1), // Jan 1 = day 1
        new Date(2024, 1, 1), // Feb 1 = day 32
      ]);
      const result = s.dt.dayofyear().data;
      expect(result[0]).toBe(1);
      expect(result[1]).toBe(32);
    });

    it("extracts quarter", () => {
      const s = new Series([
        new Date(2024, 0, 1), // Q1
        new Date(2024, 3, 1), // Q2
        new Date(2024, 6, 1), // Q3
        new Date(2024, 9, 1), // Q4
      ]);
      expect(s.dt.quarter().data).toEqual([1, 2, 3, 4]);
    });

    it("extracts weekofyear", () => {
      const s = new Series([new Date(2024, 0, 1)]); // Jan 1 2024 is Monday, week 1
      const result = s.dt.weekofyear().data;
      expect(result[0]).toBeGreaterThanOrEqual(1);
      expect(result[0]).toBeLessThanOrEqual(53);
    });

    it("checks leap year", () => {
      const s = new Series([
        new Date(2024, 0, 1), // 2024 is leap
        new Date(2023, 0, 1), // 2023 is not
        new Date(2000, 0, 1), // 2000 is leap (div by 400)
        new Date(1900, 0, 1), // 1900 is not (div by 100 but not 400)
      ]);
      expect(s.dt.is_leap_year().data).toEqual([true, false, true, false]);
    });

    it("gets days in month", () => {
      const s = new Series([
        new Date(2024, 1, 1), // Feb 2024 (leap) = 29
        new Date(2023, 1, 1), // Feb 2023 (non-leap) = 28
        new Date(2024, 0, 1), // Jan = 31
      ]);
      expect(s.dt.days_in_month().data).toEqual([29, 28, 31]);
    });
  });

  describe("null propagation", () => {
    it("propagates null through all accessors", () => {
      const s = new Series([new Date(2024, 0, 1), null]);
      expect(s.dt.year().data).toEqual([2024, null]);
      expect(s.dt.month().data).toEqual([1, null]);
      expect(s.dt.day().data).toEqual([1, null]);
      expect(s.dt.hour().data).toEqual([0, null]);
      expect(s.dt.minute().data).toEqual([0, null]);
      expect(s.dt.second().data).toEqual([0, null]);
      expect(s.dt.dayofweek().data).toEqual([0, null]); // Jan 1 2024 is Monday
      expect(s.dt.quarter().data).toEqual([1, null]);
    });
  });

  describe("conversion", () => {
    it("converts to epoch milliseconds", () => {
      const d = new Date(2024, 0, 1);
      const s = new Series([d, null]);
      expect(s.dt.timestamp().data).toEqual([d.getTime(), null]);
    });

    it("converts to ISO format", () => {
      const d = new Date(Date.UTC(2024, 0, 1, 12, 0, 0));
      const s = new Series([d]);
      expect(s.dt.isoformat().data).toEqual(["2024-01-01T12:00:00.000Z"]);
    });

    it("formats with strftime", () => {
      const d = new Date(2024, 0, 15, 9, 5, 3);
      const s = new Series([d]);
      expect(s.dt.strftime("%Y-%m-%d").data).toEqual(["2024-01-15"]);
      expect(s.dt.strftime("%H:%M:%S").data).toEqual(["09:05:03"]);
    });

    it("extracts date only", () => {
      const d = new Date(2024, 0, 15, 10, 30, 45);
      const s = new Series([d]);
      const result = s.dt.date().data[0];
      expect(result).toBeInstanceOf(Date);
      if (result instanceof Date) {
        expect(result.getHours()).toBe(0);
        expect(result.getMinutes()).toBe(0);
        expect(result.getSeconds()).toBe(0);
        expect(result.getDate()).toBe(15);
      }
    });

    it("extracts time string", () => {
      const d = new Date(2024, 0, 15, 9, 5, 3);
      const s = new Series([d]);
      expect(s.dt.time().data).toEqual(["09:05:03"]);
    });
  });

  describe("floor/ceil/round", () => {
    it("floors to day", () => {
      const d = new Date(2024, 0, 15, 10, 30, 45);
      const s = new Series([d]);
      const result = s.dt.floor("D").data[0];
      expect(result).toBeInstanceOf(Date);
      if (result instanceof Date) {
        expect(result.getHours()).toBe(0);
        expect(result.getMinutes()).toBe(0);
      }
    });

    it("floors to hour", () => {
      const d = new Date(2024, 0, 15, 10, 30, 45);
      const s = new Series([d]);
      const result = s.dt.floor("H").data[0];
      if (result instanceof Date) {
        expect(result.getHours()).toBe(10);
        expect(result.getMinutes()).toBe(0);
      }
    });

    it("ceils to hour", () => {
      const d = new Date(2024, 0, 15, 10, 30, 0);
      const s = new Series([d]);
      const result = s.dt.ceil("H").data[0];
      if (result instanceof Date) {
        expect(result.getHours()).toBe(11);
        expect(result.getMinutes()).toBe(0);
      }
    });

    it("ceil returns same if already on boundary", () => {
      const d = new Date(2024, 0, 15, 10, 0, 0);
      const s = new Series([d]);
      const result = s.dt.ceil("H").data[0];
      if (result instanceof Date) {
        expect(result.getHours()).toBe(10);
        expect(result.getMinutes()).toBe(0);
      }
    });

    it("rounds to nearest hour", () => {
      const d1 = new Date(2024, 0, 15, 10, 20, 0); // closer to 10
      const d2 = new Date(2024, 0, 15, 10, 40, 0); // closer to 11
      const s = new Series([d1, d2]);
      const result = s.dt.round("H").data;
      if (result[0] instanceof Date && result[1] instanceof Date) {
        expect(result[0].getHours()).toBe(10);
        expect(result[1].getHours()).toBe(11);
      }
    });
  });

  describe("string parsing", () => {
    it("parses ISO strings via dt accessor", () => {
      const s = new Series(["2024-01-15", "2024-06-20"]);
      expect(s.dt.year().data).toEqual([2024, 2024]);
      expect(s.dt.month().data).toEqual([1, 6]);
    });

    it("parses epoch numbers via dt accessor", () => {
      const epoch = new Date(2024, 0, 1).getTime();
      const s = new Series([epoch]);
      expect(s.dt.year().data).toEqual([2024]);
    });
  });
});

describe("to_datetime()", () => {
  it("parses string dates", () => {
    const result = to_datetime(["2024-01-01", "2024-06-15"]);
    expect(result.data[0]).toBeInstanceOf(Date);
    expect(result.data[1]).toBeInstanceOf(Date);
    expect(result.dt.year().data).toEqual([2024, 2024]);
    expect(result.dt.month().data).toEqual([1, 6]);
  });

  it("handles null values", () => {
    const result = to_datetime(["2024-01-01", null, "2024-12-31"]);
    expect(result.data[0]).toBeInstanceOf(Date);
    expect(result.data[1]).toBeNull();
    expect(result.data[2]).toBeInstanceOf(Date);
  });

  it("handles Date objects", () => {
    const d = new Date(2024, 5, 15);
    const result = to_datetime([d]);
    expect(result.data[0]).toBeInstanceOf(Date);
    expect(result.dt.month().data).toEqual([6]);
  });

  it("handles epoch numbers", () => {
    const epoch = Date.UTC(2024, 0, 1);
    const result = to_datetime([epoch]);
    expect(result.data[0]).toBeInstanceOf(Date);
  });

  it("throws on invalid date string", () => {
    expect(() => to_datetime(["not-a-date"])).toThrow();
  });

  it("accepts Series options", () => {
    const result = to_datetime(["2024-01-01"], { name: "dates", index: ["a"] });
    expect(result.name).toBe("dates");
    expect(result.index).toEqual(["a"]);
  });
});

describe("date_range()", () => {
  it("generates daily range", () => {
    const result = date_range("2024-01-01", 5, "D");
    expect(result.length).toBe(5);
    expect(result.dt.day().data).toEqual([1, 2, 3, 4, 5]);
  });

  it("generates hourly range", () => {
    const result = date_range(new Date(2024, 0, 1, 0, 0, 0), 4, "H");
    expect(result.length).toBe(4);
    expect(result.dt.hour().data).toEqual([0, 1, 2, 3]);
  });

  it("generates monthly range", () => {
    const result = date_range("2024-01-01", 3, "MS");
    expect(result.length).toBe(3);
    expect(result.dt.month().data).toEqual([1, 2, 3]);
  });

  it("generates yearly range", () => {
    const result = date_range("2020-01-01", 5, "YS");
    expect(result.length).toBe(5);
    expect(result.dt.year().data).toEqual([2020, 2021, 2022, 2023, 2024]);
  });

  it("handles 0 periods", () => {
    const result = date_range("2024-01-01", 0);
    expect(result.length).toBe(0);
  });

  it("throws for invalid periods", () => {
    expect(() => date_range("2024-01-01", -1)).toThrow();
    expect(() => date_range("2024-01-01", 1.5)).toThrow();
  });
});

describe("timedelta()", () => {
  it("computes difference in days", () => {
    const left = new Series([new Date(2024, 0, 10), new Date(2024, 0, 5)]);
    const right = new Series([new Date(2024, 0, 1), new Date(2024, 0, 1)]);
    const result = timedelta(left, right, "D");
    expect(result.data).toEqual([9, 4]);
  });

  it("computes difference in hours", () => {
    const left = new Series([new Date(2024, 0, 1, 12, 0, 0)]);
    const right = new Series([new Date(2024, 0, 1, 0, 0, 0)]);
    const result = timedelta(left, right, "H");
    expect(result.data).toEqual([12]);
  });

  it("propagates null", () => {
    const left = new Series([new Date(2024, 0, 1), null]);
    const right = new Series([new Date(2024, 0, 1), new Date(2024, 0, 1)]);
    const result = timedelta(left, right, "D");
    expect(result.data).toEqual([0, null]);
  });

  it("throws for mismatched lengths", () => {
    const left = new Series([new Date(2024, 0, 1)]);
    const right = new Series([new Date(2024, 0, 1), new Date(2024, 0, 2)]);
    expect(() => timedelta(left, right)).toThrow("lengths must match");
  });
});
