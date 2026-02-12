import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe";

/**
 * Comprehensive test suite for DataFrame join() and merge() operations.
 * Tests hash join algorithm correctness, edge cases, and all join types.
 */
describe("DataFrame.join()", () => {
  /**
   * Test inner join - only matching rows.
   */
  it("should perform inner join correctly", () => {
    // Create test DataFrames
    const left = new DataFrame({
      id: [1, 2, 3],
      name: ["Alice", "Bob", "Charlie"],
    });
    const right = new DataFrame({
      id: [1, 2, 4],
      value: [100, 200, 300],
    });

    // Perform inner join
    const result = left.join(right, "id", "inner");

    // Verify results
    expect(result.shape).toEqual([2, 3]); // 2 matching rows, 3 columns
    expect(result.columns).toEqual(["id", "name", "value"]);

    // Check data correctness
    const resultData = result.toArray();
    expect(resultData).toEqual([
      [1, "Alice", 100],
      [2, "Bob", 200],
    ]);
  });

  /**
   * Test left join - all left rows, nulls for non-matching right rows.
   */
  it("should perform left join correctly", () => {
    const left = new DataFrame({
      id: [1, 2, 3],
      name: ["Alice", "Bob", "Charlie"],
    });
    const right = new DataFrame({
      id: [1, 2, 4],
      value: [100, 200, 300],
    });

    const result = left.join(right, "id", "left");

    expect(result.shape).toEqual([3, 3]); // All 3 left rows
    expect(result.columns).toEqual(["id", "name", "value"]);

    const resultData = result.toArray();
    expect(resultData).toEqual([
      [1, "Alice", 100],
      [2, "Bob", 200],
      [3, "Charlie", null], // No match in right DataFrame
    ]);
  });

  /**
   * Test right join - all right rows, nulls for non-matching left rows.
   */
  it("should perform right join correctly", () => {
    const left = new DataFrame({
      id: [1, 2, 3],
      name: ["Alice", "Bob", "Charlie"],
    });
    const right = new DataFrame({
      id: [1, 2, 4],
      value: [100, 200, 300],
    });

    const result = left.join(right, "id", "right");

    expect(result.shape).toEqual([3, 3]); // All 3 right rows
    expect(result.columns).toEqual(["id", "name", "value"]);

    const resultData = result.toArray();
    expect(resultData[0]).toEqual([1, "Alice", 100]);
    expect(resultData[1]).toEqual([2, "Bob", 200]);
    expect(resultData[2]).toEqual([4, null, 300]); // id=4 not in left
  });

  /**
   * Test outer join - all rows from both DataFrames.
   */
  it("should perform outer join correctly", () => {
    const left = new DataFrame({
      id: [1, 2, 3],
      name: ["Alice", "Bob", "Charlie"],
    });
    const right = new DataFrame({
      id: [1, 2, 4],
      value: [100, 200, 300],
    });

    const result = left.join(right, "id", "outer");

    expect(result.shape).toEqual([4, 3]); // 4 unique ids
    expect(result.columns).toEqual(["id", "name", "value"]);

    const resultData = result.toArray();
    expect(resultData.length).toBe(4);
    // Should contain all combinations
    expect(resultData).toContainEqual([1, "Alice", 100]);
    expect(resultData).toContainEqual([2, "Bob", 200]);
    expect(resultData).toContainEqual([3, "Charlie", null]);
    expect(resultData).toContainEqual([4, null, 300]); // id=4 from right
  });

  /**
   * Test one-to-many join - multiple right rows for one left row.
   */
  it("should handle one-to-many joins", () => {
    const customers = new DataFrame({
      id: [1, 2],
      name: ["Alice", "Bob"],
    });
    const orders = new DataFrame({
      id: [1, 1, 1, 2],
      product: ["Laptop", "Mouse", "Keyboard", "Monitor"],
    });

    const result = customers.join(orders, "id", "inner");

    // Alice has 3 orders, Bob has 1 order
    expect(result.shape).toEqual([4, 3]);

    const resultData = result.toArray();
    expect(resultData).toEqual([
      [1, "Alice", "Laptop"],
      [1, "Alice", "Mouse"],
      [1, "Alice", "Keyboard"],
      [2, "Bob", "Monitor"],
    ]);
  });

  /**
   * Test many-to-many join - multiple matches on both sides.
   */
  it("should handle many-to-many joins", () => {
    const left = new DataFrame({
      id: [1, 1],
      leftVal: ["a", "b"],
    });
    const right = new DataFrame({
      id: [1, 1, 1],
      rightVal: [10, 20, 30],
    });

    const result = left.join(right, "id", "inner");
    expect(result.shape).toEqual([6, 3]);
    expect(result.toArray()).toEqual([
      [1, "a", 10],
      [1, "a", 20],
      [1, "a", 30],
      [1, "b", 10],
      [1, "b", 20],
      [1, "b", 30],
    ]);
  });

  /**
   * Test error handling - join column not found in left DataFrame.
   */
  it("should throw error if join column not in left DataFrame", () => {
    const left = new DataFrame({ a: [1, 2] });
    const right = new DataFrame({ b: [1, 2] });

    expect(() => left.join(right, "b", "inner")).toThrow(
      "Join column 'b' not found in left DataFrame"
    );
  });

  /**
   * Test error handling - join column not found in right DataFrame.
   */
  it("should throw error if join column not in right DataFrame", () => {
    const left = new DataFrame({ a: [1, 2] });
    const right = new DataFrame({ b: [1, 2] });

    expect(() => left.join(right, "a", "inner")).toThrow(
      "Join column 'a' not found in right DataFrame"
    );
  });

  /**
   * Test error handling - invalid join type.
   */
  it("should throw error for invalid join type", () => {
    const left = new DataFrame({ id: [1, 2] });
    const right = new DataFrame({ id: [1, 2] });

    // @ts-expect-error - invalid join type
    expect(() => left.join(right, "id", "sideways")).toThrow(/how must be one of/);
  });

  /**
   * Test edge case - empty DataFrames.
   */
  it("should handle empty DataFrames", () => {
    const left = new DataFrame({ id: [], name: [] });
    const right = new DataFrame({ id: [], value: [] });

    const result = left.join(right, "id", "inner");

    expect(result.shape).toEqual([0, 3]);
    expect(result.columns).toEqual(["id", "name", "value"]);
    expect(result.toArray()).toEqual([]);
  });

  /**
   * Test null join keys - nulls should not match.
   */
  it("should not match null join keys in outer joins", () => {
    const left = new DataFrame({ id: [1, null, 2], value: ["a", "b", "c"] });
    const right = new DataFrame({ id: [null, 2], score: [10, 20] });

    const result = left.join(right, "id", "outer");
    expect(result.columns).toEqual(["id", "value", "score"]);
    expect(result.toArray()).toEqual([
      [1, "a", null],
      [null, "b", null],
      [2, "c", 20],
      [null, null, 10],
    ]);
  });

  it("should not match undefined join keys in inner joins", () => {
    const left = new DataFrame({ id: [undefined, 1], value: ["skip", "ok"] });
    const right = new DataFrame({ id: [undefined, 1], score: [10, 20] });

    const result = left.join(right, "id", "inner");
    expect(result.toArray()).toEqual([[1, "ok", 20]]);
  });

  it("should handle large datasets and preserve alignment", () => {
    const leftData = {
      id: Array.from({ length: 1000 }, (_, i) => i),
      value: Array.from({ length: 1000 }, (_, i) => i * 2),
    };
    const rightData = {
      id: Array.from({ length: 1000 }, (_, i) => i),
      score: Array.from({ length: 1000 }, (_, i) => i * 3),
    };

    const left = new DataFrame(leftData);
    const right = new DataFrame(rightData);
    const result = left.join(right, "id", "inner");

    expect(result.shape).toEqual([1000, 3]);
    expect(result.iloc(0)).toEqual({ id: 0, value: 0, score: 0 });
    expect(result.iloc(999)).toEqual({ id: 999, value: 1998, score: 2997 });
  });
});

/**
 * Comprehensive test suite for DataFrame.merge() operations.
 * Tests pandas-style merge with different column names and suffix handling.
 */
describe("DataFrame.merge()", () => {
  /**
   * Test merge with same column name using 'on' parameter.
   */
  it("should merge with on parameter", () => {
    const left = new DataFrame({
      id: [1, 2, 3],
      name: ["Alice", "Bob", "Charlie"],
    });
    const right = new DataFrame({
      id: [1, 2, 4],
      value: [100, 200, 300],
    });

    const result = left.merge(right, { on: "id", how: "inner" });

    expect(result.shape).toEqual([2, 3]);
    expect(result.columns).toEqual(["id", "name", "value"]);
    expect(result.toArray()).toEqual([
      [1, "Alice", 100],
      [2, "Bob", 200],
    ]);
  });

  /**
   * Test merge with different column names using left_on/right_on.
   */
  it("should merge with different column names", () => {
    const employees = new DataFrame({
      emp_id: [1, 2, 3],
      name: ["Alice", "Bob", "Charlie"],
    });
    const salaries = new DataFrame({
      employee_id: [1, 2, 4],
      salary: [50000, 60000, 55000],
    });

    const result = employees.merge(salaries, {
      left_on: "emp_id",
      right_on: "employee_id",
      how: "inner",
    });

    expect(result.shape).toEqual([2, 4]); // 2 rows, 4 columns (emp_id, name, employee_id, salary)
    expect(result.columns).toContain("emp_id");
    expect(result.columns).toContain("name");
    expect(result.columns).toContain("employee_id");
    expect(result.columns).toContain("salary");

    const resultData = result.toArray();
    expect(resultData).toEqual([
      [1, "Alice", 1, 50000],
      [2, "Bob", 2, 60000],
    ]);
  });

  /**
   * Test many-to-many merge - multiple matches on both sides.
   */
  it("should handle many-to-many merges", () => {
    const left = new DataFrame({
      id: [1, 1],
      leftVal: ["a", "b"],
    });
    const right = new DataFrame({
      id: [1, 1, 1],
      rightVal: [10, 20, 30],
    });

    const result = left.merge(right, { on: "id", how: "inner" });
    expect(result.shape).toEqual([6, 3]);
    expect(result.toArray()).toEqual([
      [1, "a", 10],
      [1, "a", 20],
      [1, "a", 30],
      [1, "b", 10],
      [1, "b", 20],
      [1, "b", 30],
    ]);
  });

  /**
   * Test null merge keys - nulls should not match.
   */
  it("should not match null/undefined merge keys in inner merges", () => {
    const left = new DataFrame({
      id: [1, null, undefined],
      value: ["a", "b", "c"],
    });
    const right = new DataFrame({
      id: [undefined, 1, null],
      score: [10, 20, 30],
    });

    const result = left.merge(right, { on: "id", how: "inner" });
    expect(result.toArray()).toEqual([[1, "a", 20]]);
  });

  it("should match NaN merge keys in inner merges", () => {
    const left = new DataFrame({ id: [NaN, 1], value: ["a", "b"] });
    const right = new DataFrame({ id: [NaN, 2], score: [10, 20] });

    const result = left.merge(right, { on: "id", how: "inner" });
    expect(result.columns).toEqual(["id", "value", "score"]);
    expect(result.shape).toEqual([1, 3]);
    const row = result.toArray()[0];
    expect(Number.isNaN(Number(row?.[0]))).toBe(true);
    expect(row?.[1]).toBe("a");
    expect(row?.[2]).toBe(10);
  });

  /**
   * Test column name conflicts with suffixes.
   */
  it("should handle column name conflicts with suffixes", () => {
    const left = new DataFrame({
      id: [1, 2],
      value: [10, 20],
    });
    const right = new DataFrame({
      id: [1, 2],
      value: [100, 200],
    });

    const result = left.merge(right, { on: "id", how: "inner" });

    // Should add default suffixes _x and _y
    expect(result.columns).toEqual(["id", "value_x", "value_y"]);
    expect(result.toArray()).toEqual([
      [1, 10, 100],
      [2, 20, 200],
    ]);
  });

  /**
   * Test custom suffixes.
   */
  it("should support custom suffixes", () => {
    const left = new DataFrame({
      id: [1, 2],
      score: [10, 20],
    });
    const right = new DataFrame({
      id: [1, 2],
      score: [100, 200],
    });

    const result = left.merge(right, {
      on: "id",
      how: "inner",
      suffixes: ["_left", "_right"],
    });

    expect(result.columns).toEqual(["id", "score_left", "score_right"]);
    expect(result.toArray()).toEqual([
      [1, 10, 100],
      [2, 20, 200],
    ]);
  });

  /**
   * Test left merge behavior.
   */
  it("should perform left merge correctly", () => {
    const left = new DataFrame({
      id: [1, 2, 3],
      name: ["Alice", "Bob", "Charlie"],
    });
    const right = new DataFrame({
      id: [1, 2],
      value: [100, 200],
    });

    const result = left.merge(right, { on: "id", how: "left" });

    expect(result.shape).toEqual([3, 3]);

    const resultData = result.toArray();
    expect(resultData[2]).toContain("Charlie");
    expect(resultData[2]).toContain(null); // No matching value
  });

  /**
   * Test error handling - conflicting parameters.
   */
  it("should throw error for conflicting parameters", () => {
    const left = new DataFrame({ id: [1, 2] });
    const right = new DataFrame({ id: [1, 2] });

    expect(() => left.merge(right, { on: "id", left_on: "id" })).toThrow(
      'Cannot specify both "on" and "left_on"/"right_on"'
    );
  });

  /**
   * Test error handling - missing required parameters.
   */
  it("should throw error for missing parameters", () => {
    const left = new DataFrame({ id: [1, 2] });
    const right = new DataFrame({ id: [1, 2] });

    expect(() => left.merge(right, {})).toThrow(
      'Must specify either "on" or both "left_on" and "right_on"'
    );
  });

  /**
   * Test error handling - invalid how parameter.
   */
  it("should throw error for invalid how value", () => {
    const left = new DataFrame({ id: [1, 2] });
    const right = new DataFrame({ id: [1, 2] });

    // @ts-expect-error - invalid how value
    expect(() => left.merge(right, { on: "id", how: "sideways" })).toThrow(/how must be one of/);
  });

  /**
   * Test error handling - column not found in left DataFrame.
   */
  it("should throw error if left column not found", () => {
    const left = new DataFrame({ a: [1, 2] });
    const right = new DataFrame({ b: [1, 2] });

    expect(() => left.merge(right, { left_on: "b", right_on: "b" })).toThrow(
      "Column 'b' not found in left DataFrame"
    );
  });

  /**
   * Test error handling - column not found in right DataFrame.
   */
  it("should throw error if right column not found", () => {
    const left = new DataFrame({ a: [1, 2] });
    const right = new DataFrame({ b: [1, 2] });

    expect(() => left.merge(right, { left_on: "a", right_on: "a" })).toThrow(
      "Column 'a' not found in right DataFrame"
    );
  });
});
