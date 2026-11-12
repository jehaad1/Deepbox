import { describe, expect, it } from "vitest";
import { Series, toDatetime } from "../../src/dataframe/index";

describe("dt.isoformat uses the same naive wall-clock fields as the other accessors (DF-5)", () => {
  // pandas: Timestamp('2024-01-01 12:00:00.5').isoformat() -> '2024-01-01T12:00:00.500000'
  //         Timestamp('2024-01-01 12:00:00').isoformat()   -> '2024-01-01T12:00:00'
  //         Timestamp('2024-02-29 23:59:59.999').isoformat() -> '2024-02-29T23:59:59.999000'
  it("matches pandas formatting for fractional and whole seconds", () => {
    const s = new Series([
      "2024-01-01 12:00:00.5",
      "2024-01-01 12:00:00",
      "2024-02-29 23:59:59.999",
      "2024-01-01",
    ]);
    expect(s.dt.isoformat().data).toEqual([
      "2024-01-01T12:00:00.500000",
      "2024-01-01T12:00:00",
      "2024-02-29T23:59:59.999000",
      "2024-01-01T00:00:00",
    ]);
  });

  it("agrees with hour(), minute() and second() in any time zone", () => {
    const d = new Date(2024, 4, 17, 8, 5, 9, 40);
    const s = new Series([d]);
    const iso = s.dt.isoformat().data[0] as string;
    const hour = s.dt.hour().data[0];
    const minute = s.dt.minute().data[0];
    const second = s.dt.second().data[0];
    expect(iso).toBe(
      `2024-05-17T${String(hour).padStart(2, "0")}:${String(minute).padStart(2, "0")}:${String(second).padStart(2, "0")}.040000`
    );
    expect(iso).toBe("2024-05-17T08:05:09.040000");
  });

  it("round-trips through toDatetime and keeps nulls", () => {
    const s = toDatetime(["2024-03-05 01:02:03.25", null]);
    const iso = s.dt.isoformat();
    expect(iso.data).toEqual(["2024-03-05T01:02:03.250000", null]);
    expect(toDatetime(iso.data).data[0]?.getTime()).toBe(s.data[0]?.getTime());
  });
});
