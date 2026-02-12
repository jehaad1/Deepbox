import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe";

describe("DataFrame IO error branches", () => {
  it("wraps CSV/JSON read errors", async () => {
    await expect(DataFrame.readCsv("/tmp/deepbox-missing-dir/read-missing.csv")).rejects.toThrow(
      /Failed to read CSV file/i
    );
    await expect(DataFrame.readJson("/tmp/deepbox-missing-dir/read-missing.json")).rejects.toThrow(
      /Failed to read JSON file/i
    );
  });

  it("wraps CSV/JSON write errors", async () => {
    const df = new DataFrame({ a: [1] });
    await expect(df.toCsv("/tmp/deepbox-missing-dir/write-missing.csv")).rejects.toThrow(
      /Failed to write CSV file/i
    );
    await expect(df.toJson("/tmp/deepbox-missing-dir/write-missing.json")).rejects.toThrow(
      /Failed to write JSON file/i
    );
  });
});
