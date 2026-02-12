import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe";

/**
 * Comprehensive test suite for DataFrame CSV operations.
 * Tests parsing, generation, quoting, and edge cases.
 */
describe("DataFrame CSV Operations", () => {
  /**
   * Test basic CSV parsing with headers.
   */
  describe("fromCsvString()", () => {
    it("should parse simple CSV with headers", () => {
      const csv = "name,age,city\nAlice,25,NYC\nBob,30,LA";
      const df = DataFrame.fromCsvString(csv);

      expect(df.shape).toEqual([2, 3]);
      expect(df.columns).toEqual(["name", "age", "city"]);

      const data = df.toArray();
      expect(data[0]).toEqual(["Alice", 25, "NYC"]);
      expect(data[1]).toEqual(["Bob", 30, "LA"]);
    });

    /**
     * Test CSV parsing without headers (auto-generated column names).
     */
    it("should handle CSV without headers", () => {
      const csv = "1,2,3\n4,5,6";
      const df = DataFrame.fromCsvString(csv, { hasHeader: false });

      expect(df.columns).toEqual(["col0", "col1", "col2"]);
      expect(df.shape).toEqual([2, 3]);
    });

    /**
     * Test quoted fields with commas.
     */
    it("should handle quoted fields with commas", () => {
      const csv = 'name,description\nAlice,"Hello, World"\nBob,"Test, Data"';
      const df = DataFrame.fromCsvString(csv);

      const data = df.toArray();
      expect(data[0]).toEqual(["Alice", "Hello, World"]);
      expect(data[1]).toEqual(["Bob", "Test, Data"]);
    });

    /**
     * Test escaped quotes (double quotes).
     */
    it("should handle escaped quotes", () => {
      const csv = 'name,quote\nAlice,"He said ""Hello"""\nBob,"She said ""Hi"""';
      const df = DataFrame.fromCsvString(csv);

      const data = df.toArray();
      expect(data[0][1]).toBe('He said "Hello"');
      expect(data[1][1]).toBe('She said "Hi"');
    });

    /**
     * Test custom delimiter.
     */
    it("should support custom delimiter", () => {
      const csv = "name;age;city\nAlice;25;NYC\nBob;30;LA";
      const df = DataFrame.fromCsvString(csv, { delimiter: ";" });

      expect(df.columns).toEqual(["name", "age", "city"]);
      expect(df.shape).toEqual([2, 3]);
      expect(df.toArray()).toEqual([
        ["Alice", 25, "NYC"],
        ["Bob", 30, "LA"],
      ]);
    });

    /**
     * Test skipping rows.
     */
    it("should skip rows from beginning", () => {
      const csv = "comment\ncomment\nname,age\nAlice,25\nBob,30";
      const df = DataFrame.fromCsvString(csv, { skipRows: 2 });

      expect(df.columns).toEqual(["name", "age"]);
      expect(df.shape).toEqual([2, 2]);
      expect(df.toArray()).toEqual([
        ["Alice", 25],
        ["Bob", 30],
      ]);
    });

    /**
     * Test type inference - numbers.
     */
    it("should infer numeric types", () => {
      const csv = "name,age,score\nAlice,25,98.5\nBob,30,87.3";
      const df = DataFrame.fromCsvString(csv);

      const ageCol = df
        .select(["age"])
        .toArray()
        .map((row) => row[0]);
      const scoreCol = df
        .select(["score"])
        .toArray()
        .map((row) => row[0]);

      expect(typeof ageCol[0]).toBe("number");
      expect(typeof scoreCol[0]).toBe("number");
      expect(ageCol[0]).toBe(25);
      expect(scoreCol[0]).toBe(98.5);
    });

    /**
     * Test type inference - booleans.
     */
    it("should infer boolean types", () => {
      const csv = "name,active\nAlice,true\nBob,false";
      const df = DataFrame.fromCsvString(csv);

      const activeCol = df
        .select(["active"])
        .toArray()
        .map((row) => row[0]);
      expect(activeCol[0]).toBe(true);
      expect(activeCol[1]).toBe(false);
    });

    /**
     * Test null value handling.
     */
    it("should handle null values", () => {
      const csv = "name,age\nAlice,25\nBob,\nCharlie,null";
      const df = DataFrame.fromCsvString(csv);

      const ageCol = df
        .select(["age"])
        .toArray()
        .map((row) => row[0]);
      expect(ageCol[1]).toBe(null);
      expect(ageCol[2]).toBe(null);
    });

    /**
     * Test empty CSV error.
     */
    it("should throw error for empty CSV", () => {
      expect(() => DataFrame.fromCsvString("")).toThrow("CSV contains no data rows");
    });

    /**
     * Test inconsistent row length error.
     */
    it("should throw error for inconsistent row lengths", () => {
      const csv = "a,b,c\n1,2,3\n4,5"; // Second row missing column

      expect(() => DataFrame.fromCsvString(csv)).toThrow(/Row \d+ has \d+ fields, expected \d+/);
    });

    /**
     * Test unmatched quote error.
     */
    it("should throw error for unmatched quotes", () => {
      const csv = 'a,b\n"unterminated,1\n2,3';
      expect(() => DataFrame.fromCsvString(csv)).toThrow(/unmatched quote/);
    });

    /**
     * Test duplicate header error.
     */
    it("should throw error for duplicate headers", () => {
      const csv = "a,a\n1,2";
      expect(() => DataFrame.fromCsvString(csv)).toThrow(/Duplicate column name/);
    });

    /**
     * Test empty lines are skipped.
     */
    it("should skip empty lines", () => {
      const csv = "name,age\nAlice,25\n\nBob,30\n";
      const df = DataFrame.fromCsvString(csv);

      expect(df.shape).toEqual([2, 2]);
      expect(df.toArray()).toEqual([
        ["Alice", 25],
        ["Bob", 30],
      ]);
    });

    /**
     * Test newlines in quoted fields.
     */
    it("should handle newlines in quoted fields", () => {
      const csv = 'name,description\nAlice,"Line 1\nLine 2"\nBob,"Single"';
      const df = DataFrame.fromCsvString(csv);

      const descSeries = df.get("description");
      expect(descSeries.data[0]).toBe("Line 1\nLine 2");
      expect(descSeries.data[1]).toBe("Single");
    });
  });

  /**
   * Test CSV string generation.
   */
  describe("toCsvString()", () => {
    it("should generate simple CSV", () => {
      const df = new DataFrame({
        name: ["Alice", "Bob"],
        age: [25, 30],
      });

      const csv = df.toCsvString();
      const lines = csv.split("\n");

      expect(lines[0]).toBe("name,age");
      expect(lines[1]).toBe("Alice,25");
      expect(lines[2]).toBe("Bob,30");
    });

    /**
     * Test quoting fields with commas.
     */
    it("should quote fields containing delimiter", () => {
      const df = new DataFrame({
        name: ["Alice", "Bob"],
        description: ["Hello, World", "Test"],
      });

      const csv = df.toCsvString();

      expect(csv).toContain('"Hello, World"');
      expect(csv).not.toContain('"Test"'); // No quotes needed
    });

    /**
     * Test escaping quotes.
     */
    it("should escape quotes by doubling them", () => {
      const df = new DataFrame({
        name: ["Alice"],
        quote: ['He said "Hello"'],
      });

      const csv = df.toCsvString();

      expect(csv).toContain('"He said ""Hello"""');
    });

    /**
     * Test custom delimiter.
     */
    it("should support custom delimiter", () => {
      const df = new DataFrame({
        name: ["Alice", "Bob"],
        age: [25, 30],
      });

      const csv = df.toCsvString({ delimiter: ";" });

      expect(csv).toContain("name;age");
      expect(csv).toContain("Alice;25");
    });

    /**
     * Test including index.
     */
    it("should include index when requested", () => {
      const df = new DataFrame(
        {
          name: ["Alice", "Bob"],
          age: [25, 30],
        },
        { index: ["id1", "id2"] }
      );

      const csv = df.toCsvString({ includeIndex: true });
      const lines = csv.split("\n");

      expect(lines[0]).toBe("index,name,age");
      expect(lines[1]).toContain("id1");
      expect(lines[2]).toContain("id2");
    });

    /**
     * Test without header.
     */
    it("should omit header when requested", () => {
      const df = new DataFrame({
        name: ["Alice", "Bob"],
        age: [25, 30],
      });

      const csv = df.toCsvString({ header: false });

      expect(csv).not.toContain("name");
      expect(csv).not.toContain("age");
      expect(csv).toBe("Alice,25\nBob,30");
    });

    /**
     * Test null value handling.
     */
    it("should represent null as empty string", () => {
      const df = new DataFrame({
        name: ["Alice", "Bob"],
        age: [25, null],
      });

      const csv = df.toCsvString();
      const lines = csv.split("\n");

      expect(lines[2]).toBe("Bob,");
    });

    /**
     * Test empty DataFrame.
     */
    it("should handle empty DataFrame", () => {
      const df = new DataFrame({ a: [], b: [] });
      const csv = df.toCsvString();

      expect(csv).toBe("a,b");
    });
  });

  /**
   * Test round-trip consistency (CSV -> DataFrame -> CSV).
   */
  describe("Round-trip consistency", () => {
    it("should maintain data through parse and generate cycle", () => {
      const original = new DataFrame({
        name: ["Alice", "Bob", "Charlie"],
        age: [25, 30, 35],
        city: ["NYC", "LA", "Chicago"],
      });

      const csv = original.toCsvString();
      const parsed = DataFrame.fromCsvString(csv);

      expect(parsed.shape).toEqual(original.shape);
      expect(parsed.columns).toEqual(original.columns);
      expect(parsed.toArray()).toEqual(original.toArray());
    });

    /**
     * Test with complex data (quotes, commas, nulls).
     */
    it("should handle complex data round-trip", () => {
      const original = new DataFrame({
        name: ["Alice", "Bob"],
        description: ['Hello, "World"', null],
        score: [98.5, 87.3],
      });

      const csv = original.toCsvString();
      const parsed = DataFrame.fromCsvString(csv);

      expect(parsed.toArray()).toEqual(original.toArray());
    });

    it("should round-trip includeIndex with custom delimiter", () => {
      const original = new DataFrame(
        { name: ["Alice", "Bob"], age: [25, 30] },
        { index: ["row1", "row2"] }
      );

      const csv = original.toCsvString({ includeIndex: true, delimiter: ";" });
      const parsed = DataFrame.fromCsvString(csv, { delimiter: ";" });

      expect(parsed.columns).toEqual(["index", "name", "age"]);
      expect(parsed.toArray()).toEqual([
        ["row1", "Alice", 25],
        ["row2", "Bob", 30],
      ]);
    });
  });
});
