import { beforeEach, describe, expect, it } from "vitest";
import { DataFrame, Series } from "../../src/dataframe";
import { tensor } from "../../src/ndarray";
import { figure, gca, kdeplot } from "../../src/plot";
import { gaussianKde } from "../../src/stats";

/** Columns, index and cell values of a frame, to compare two results. */
const snapshot = (df: DataFrame): unknown => ({
  columns: [...df.columns],
  index: [...df.index],
  values: df.toArray(),
});

const frame = (): DataFrame =>
  new DataFrame(
    {
      id: [1, 1, 2, 3, 3],
      grp: ["a", "a", "b", "b", "b"],
      x: [10, 10, 20, 40, 50],
      y: [1.5, 1.5, 2.5, Number.NaN, 4],
    },
    { index: ["r0", "r1", "r2", "r3", "r4"] }
  );

describe("DataFrame camelCase methods match the snake_case ones", () => {
  it("dropDuplicates", () => {
    const df = frame();
    expect(snapshot(df.dropDuplicates())).toEqual(snapshot(df.drop_duplicates()));
    expect(snapshot(df.dropDuplicates(["id"], "last"))).toEqual(
      snapshot(df.drop_duplicates(["id"], "last"))
    );
    expect(snapshot(df.dropDuplicates(["grp"], false))).toEqual(
      snapshot(df.drop_duplicates(["grp"], false))
    );
    expect(df.dropDuplicates().shape[0]).toBe(4);
  });

  it("resetIndex", () => {
    const df = frame();
    expect(snapshot(df.resetIndex())).toEqual(snapshot(df.reset_index()));
    expect(snapshot(df.resetIndex(true))).toEqual(snapshot(df.reset_index(true)));
    const clash = new DataFrame({ index: [1, 2] }, { index: ["p", "q"] });
    expect(snapshot(clash.resetIndex())).toEqual(snapshot(clash.reset_index()));
  });

  it("setIndex", () => {
    const df = new DataFrame({ id: ["a", "b", "c"], v: [1, 2, 3] });
    expect(snapshot(df.setIndex("id"))).toEqual(snapshot(df.set_index("id")));
    expect(snapshot(df.setIndex("id", false))).toEqual(snapshot(df.set_index("id", false)));
    expect(() => df.setIndex("nope")).toThrow();
    expect(() => df.set_index("nope")).toThrow();
  });

  it("pctChange", () => {
    const df = new DataFrame({ a: [100, 110, 121, 0, 5], b: [1, 2, 4, 8, 16] });
    expect(snapshot(df.pctChange())).toEqual(snapshot(df.pct_change()));
    expect(snapshot(df.pctChange(2))).toEqual(snapshot(df.pct_change(2)));
    expect(() => df.pctChange(-1)).toThrow();
    expect(() => df.pct_change(-1)).toThrow();
  });

  it("memoryUsage", () => {
    const df = frame();
    expect(snapshot(df.memoryUsage())).toEqual(snapshot(df.memory_usage()));
    expect(df.memoryUsage().columns).toEqual(["column", "bytes"]);
  });

  it("pivotTable", () => {
    const df = new DataFrame({
      r: ["a", "a", "b", "b", "b"],
      c: ["u", "v", "u", "u", "v"],
      v: [1, 2, 3, 4, 5],
    });
    const options = { index: "r", columns: "c", values: "v" } as const;
    expect(snapshot(df.pivotTable(options))).toEqual(snapshot(df.pivot_table(options)));
    for (const aggFunc of ["sum", "count", "min", "max", "median", "first", "last"] as const) {
      expect(snapshot(df.pivotTable({ ...options, aggFunc }))).toEqual(
        snapshot(df.pivot_table({ ...options, aggFunc }))
      );
    }
    expect(() => df.pivotTable({ ...options, values: "missing" })).toThrow();
  });

  it("valueCounts and value_counts", () => {
    const df = frame();
    expect(snapshot(df.valueCounts("grp"))).toEqual(snapshot(df.value_counts("grp")));
    expect(snapshot(df.valueCounts("grp", "id", { normalize: true, ascending: true }))).toEqual(
      snapshot(df.value_counts("grp", "id", { normalize: true, ascending: true }))
    );
    expect(() => df.valueCounts("grp", { dropna: true }, "id")).toThrow(/valueCounts takes/);
  });
});

describe("DataFrame.merge accepts leftOn and rightOn", () => {
  const employees = (): DataFrame =>
    new DataFrame({ emp_id: [1, 2, 3], name: ["Alice", "Bob", "Charlie"] });
  const salaries = (): DataFrame =>
    new DataFrame({ employee_id: [1, 2, 4], salary: [50000, 60000, 55000] });

  it("leftOn and rightOn give the same result as left_on and right_on", () => {
    for (const how of ["inner", "left", "right", "outer"] as const) {
      const camel = employees().merge(salaries(), {
        leftOn: "emp_id",
        rightOn: "employee_id",
        how,
      });
      const snake = employees().merge(salaries(), {
        left_on: "emp_id",
        right_on: "employee_id",
        how,
      });
      expect(snapshot(camel)).toEqual(snapshot(snake));
    }
  });

  it("the snake_case keys still work on their own", () => {
    const out = employees().merge(salaries(), { left_on: "emp_id", right_on: "employee_id" });
    expect(out.shape[0]).toBe(2);
    expect(out.columns).toEqual(["emp_id", "name", "employee_id", "salary"]);
  });

  it("camelCase wins when both spellings are given", () => {
    const expected = employees().merge(salaries(), { leftOn: "emp_id", rightOn: "employee_id" });
    const both = employees().merge(salaries(), {
      leftOn: "emp_id",
      rightOn: "employee_id",
      left_on: "name",
      right_on: "salary",
    });
    expect(snapshot(both)).toEqual(snapshot(expected));
    const mixed = employees().merge(salaries(), { leftOn: "emp_id", right_on: "employee_id" });
    expect(snapshot(mixed)).toEqual(snapshot(expected));
  });

  it("validates the keys under both spellings", () => {
    expect(() => employees().merge(salaries(), { on: "emp_id", leftOn: "emp_id" })).toThrow(
      /Cannot specify both/
    );
    expect(() => employees().merge(salaries(), { on: "emp_id", rightOn: "emp_id" })).toThrow(
      /Cannot specify both/
    );
    expect(() => employees().merge(salaries(), { leftOn: "emp_id" })).toThrow(/Must specify/);
    expect(() => employees().merge(salaries(), { rightOn: "employee_id" })).toThrow(/Must specify/);
    expect(() =>
      employees().merge(salaries(), { leftOn: "missing", rightOn: "employee_id" })
    ).toThrow("Column 'missing' not found in left DataFrame");
    expect(() => employees().merge(salaries(), { leftOn: "emp_id", rightOn: "missing" })).toThrow(
      "Column 'missing' not found in right DataFrame"
    );
  });
});

describe("DateTimeAccessor camelCase methods match the old names", () => {
  const dates = (): Series<Date | null> =>
    new Series<Date | null>(
      [
        new Date(2024, 0, 1),
        new Date(2024, 1, 29),
        new Date(2024, 2, 31),
        new Date(2024, 5, 30),
        new Date(2023, 1, 28),
        new Date(2023, 11, 31),
        new Date(2021, 0, 3),
        new Date(2020, 11, 31),
        null,
        new Date(2024, 8, 30),
        new Date(2024, 9, 1),
      ],
      { name: "d" }
    );

  const same = (camel: Series<unknown>, old: Series<unknown>): void => {
    expect(camel.toArray()).toEqual(old.toArray());
    expect([...camel.index]).toEqual([...old.index]);
    expect(camel.name).toBe(old.name);
  };

  it("matches for every renamed accessor method", () => {
    const dt = dates().dt;
    same(dt.isLeapYear(), dt.is_leap_year());
    same(dt.daysInMonth(), dt.days_in_month());
    same(dt.daysInMonth(), dt.daysinmonth());
    same(dt.isMonthStart(), dt.is_month_start());
    same(dt.isMonthEnd(), dt.is_month_end());
    same(dt.isQuarterStart(), dt.is_quarter_start());
    same(dt.isQuarterEnd(), dt.is_quarter_end());
    same(dt.isYearStart(), dt.is_year_start());
    same(dt.isYearEnd(), dt.is_year_end());
    same(dt.dayName(), dt.day_name());
    same(dt.monthName(), dt.month_name());
    same(dt.dayOfWeek(), dt.dayofweek());
    same(dt.dayOfYear(), dt.dayofyear());
    same(dt.weekOfYear(), dt.weekofyear());
  });

  it("returns the expected values", () => {
    const dt = dates().dt;
    expect(dt.isLeapYear().toArray()).toEqual([
      true,
      true,
      true,
      true,
      false,
      false,
      false,
      true,
      null,
      true,
      true,
    ]);
    expect(dt.daysInMonth().toArray().slice(0, 5)).toEqual([31, 29, 31, 30, 28]);
    expect(dt.dayName().toArray().slice(0, 2)).toEqual(["Monday", "Thursday"]);
    expect(dt.monthName().toArray().slice(0, 2)).toEqual(["January", "February"]);
    expect(dt.dayOfWeek().toArray().slice(0, 2)).toEqual([0, 3]);
    expect(dt.dayOfYear().toArray().slice(0, 2)).toEqual([1, 60]);
    expect(dt.weekOfYear().toArray()[6]).toBe(53);
    expect(dt.isMonthStart().toArray()[0]).toBe(true);
    expect(dt.isMonthEnd().toArray()[2]).toBe(true);
    expect(dt.isQuarterStart().toArray()[10]).toBe(true);
    expect(dt.isQuarterEnd().toArray()[9]).toBe(true);
    expect(dt.isYearStart().toArray()[0]).toBe(true);
    expect(dt.isYearEnd().toArray()[5]).toBe(true);
  });

  it("weekday and week stay aliases of the new names", () => {
    const dt = dates().dt;
    same(dt.weekday(), dt.dayOfWeek());
    same(dt.week(), dt.weekOfYear());
  });
});

describe("StringAccessor.getDummies", () => {
  it("matches get_dummies", () => {
    const s = new Series<string | null>(["a|b", "b|c", null, "a", ""], { name: "tags" });
    expect(snapshot(s.str.getDummies())).toEqual(snapshot(s.str.get_dummies()));
    const t = new Series<string | null>(["a,b", "b", null]);
    expect(snapshot(t.str.getDummies(","))).toEqual(snapshot(t.str.get_dummies(",")));
    expect(snapshot(t.str.getDummies(""))).toEqual(snapshot(t.str.get_dummies("")));
    expect(() => t.str.getDummies(1 as unknown as string)).toThrow();
  });
});

describe("StyleAccessor camelCase methods match the snake_case ones", () => {
  const data = (): DataFrame =>
    new DataFrame({ a: [1, 5, 3, 5], b: [4, Number.NaN, 6, 4], c: ["x", "y", "z", "w"] });

  it("highlightMax, highlightMin and highlightNull", () => {
    const df = data();
    expect(df.style.highlightMax().toHTML()).toBe(df.style.highlight_max().toHTML());
    expect(df.style.highlightMax({ color: "red" }).toHTML()).toBe(
      df.style.highlight_max({ color: "red" }).toHTML()
    );
    expect(df.style.highlightMin().toHTML()).toBe(df.style.highlight_min().toHTML());
    expect(df.style.highlightMin({ color: "blue" }).toHTML()).toBe(
      df.style.highlight_min({ color: "blue" }).toHTML()
    );
    expect(df.style.highlightNull().toHTML()).toBe(df.style.highlight_null().toHTML());
    expect(df.style.highlightNull({ color: "gray" }).toHTML()).toBe(
      df.style.highlight_null({ color: "gray" }).toHTML()
    );
    expect(df.style.highlightMax().toHTML()).toContain("background:#d4edda");
  });

  it("backgroundGradient", () => {
    const df = data();
    expect(df.style.backgroundGradient().toHTML()).toBe(df.style.background_gradient().toHTML());
    expect(df.style.backgroundGradient("#fff", "#f00").toHTML()).toBe(
      df.style.background_gradient("#fff", "#f00").toHTML()
    );
    expect(() => df.style.backgroundGradient("red")).toThrow();
    expect(() => df.style.background_gradient("red")).toThrow();
  });

  it("the camelCase methods chain and return the accessor", () => {
    const style = data().style;
    expect(style.highlightMax()).toBe(style);
    expect(style.highlightMin()).toBe(style);
    expect(style.highlightNull()).toBe(style);
    expect(style.backgroundGradient()).toBe(style);
    expect(style.highlight_max()).toBe(style);
  });
});

describe("snake_case option keys still work and camelCase wins", () => {
  const sample = [1, 2, 3, 4, 5, 7, 9];

  it("gaussianKde bwMethod and bw_method", () => {
    const points = [0, 2, 4, 6];
    const camel = gaussianKde(sample, { bwMethod: 0.8 }).evaluate(points);
    const snake = gaussianKde(sample, { bw_method: 0.8 }).evaluate(points);
    expect(Array.from(snake)).toEqual(Array.from(camel));
    const both = gaussianKde(sample, { bwMethod: 0.8, bw_method: 3 }).evaluate(points);
    expect(Array.from(both)).toEqual(Array.from(camel));
    const named = gaussianKde(sample, { bwMethod: "silverman", bw_method: 3 }).evaluate(points);
    expect(Array.from(named)).toEqual(
      Array.from(gaussianKde(sample, { bw_method: "silverman" }).evaluate(points))
    );
  });

  type Drawable = { readonly x: Float64Array; readonly y: Float64Array };
  const drawn = (options: Parameters<typeof kdeplot>[1]): { x: number[]; y: number[] } => {
    figure();
    kdeplot(tensor(sample), { gridSize: 16, ...options });
    const drawables = (gca() as unknown as { drawables: Drawable[] }).drawables;
    const line = drawables[0];
    if (!line) throw new Error("no line drawn");
    return { x: Array.from(line.x), y: Array.from(line.y) };
  };

  beforeEach(() => {
    figure();
  });

  it("kdeplot bwMethod and bw_method", () => {
    const camel = drawn({ bwMethod: 0.8 });
    expect(drawn({ bw_method: 0.8 })).toEqual(camel);
    expect(drawn({ bwMethod: 0.8, bw_method: 3 })).toEqual(camel);
    expect(drawn({ bwMethod: "silverman" })).toEqual(drawn({ bw_method: "silverman" }));
    expect(drawn({ bwMethod: 0.8 })).not.toEqual(drawn({ bwMethod: 3 }));
    expect(drawn({})).toEqual(drawn({ bwMethod: "scott" }));
  });
});
