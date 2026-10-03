/**
 * Example 47: DataFrame IO & Styling
 *
 * DataFrame input and output and report formatting: CSV, JSON, XLSX and Parquet
 * round trips, date parsing, the string accessor, df.style (HTML and ANSI output),
 * and df.plot for an SVG line chart. Files are written to the output folder.
 */

import { mkdir, writeFile } from "node:fs/promises";
import {
  DataFrame,
  dateRange,
  readParquet,
  readXlsx,
  toDatetime,
  writeParquet,
  writeXlsx,
} from "deepbox/dataframe";

const OUTPUT_DIR = "docs/examples/47-dataframe-io-styling/output";

console.log("=".repeat(60));
console.log("Example 47: DataFrame IO & Styling");
console.log("=".repeat(60));

await mkdir(OUTPUT_DIR, { recursive: true });

const orderDates = dateRange("2026-03-01", 6, "D");
const orderDateStrings = Array.from(orderDates.data, (value) =>
  value instanceof Date ? value.toISOString().slice(0, 10) : ""
);

const rows = [
  {
    orderDate: orderDateStrings[0] ?? "2026-03-01",
    region: "GCC",
    segment: "Enterprise",
    revenue: 180_000,
    supportTickets: 12,
    activeCampaign: true,
  },
  {
    orderDate: orderDateStrings[1] ?? "2026-03-02",
    region: "North Africa",
    segment: "SMB",
    revenue: 124_000,
    supportTickets: 21,
    activeCampaign: false,
  },
  {
    orderDate: orderDateStrings[2] ?? "2026-03-03",
    region: "GCC",
    segment: "Mid-Market",
    revenue: 155_000,
    supportTickets: 16,
    activeCampaign: true,
  },
  {
    orderDate: orderDateStrings[3] ?? "2026-03-04",
    region: "Levant",
    segment: "Enterprise",
    revenue: 212_000,
    supportTickets: 9,
    activeCampaign: true,
  },
  {
    orderDate: orderDateStrings[4] ?? "2026-03-05",
    region: "North Africa",
    segment: "Enterprise",
    revenue: 167_000,
    supportTickets: 13,
    activeCampaign: false,
  },
  {
    orderDate: orderDateStrings[5] ?? "2026-03-06",
    region: "Levant",
    segment: "SMB",
    revenue: 98_000,
    supportTickets: 24,
    activeCampaign: false,
  },
];

const df = new DataFrame({
  orderDate: rows.map((row) => row.orderDate),
  region: rows.map((row) => row.region),
  segment: rows.map((row) => row.segment),
  revenue: rows.map((row) => row.revenue),
  supportTickets: rows.map((row) => row.supportTickets),
  activeCampaign: rows.map((row) => row.activeCampaign),
});

// ============================================================================
// Part 1: CSV round trip and date parsing
// ============================================================================
console.log("\nPart 1: CSV and Date Parsing");
console.log("-".repeat(60));

const csvRoundTrip = DataFrame.fromCsvString(df.toCsvString());
const parsedDates = toDatetime(rows.map((row) => row.orderDate));

console.log(
  `Round-trip CSV shape: ${csvRoundTrip.shape[0]} rows x ${csvRoundTrip.shape[1]} columns`
);
// dayOfWeek() numbers the days from Monday = 0 to Sunday = 6
console.log(
  `Parsed weekday numbers: ${Array.from(parsedDates.dt.dayOfWeek().data, (value) => String(value)).join(", ")}`
);

// ============================================================================
// Part 2: JSON, XLSX and Parquet round trips
// ============================================================================
console.log("\nPart 2: JSON / XLSX / Parquet");
console.log("-".repeat(60));

const jsonPath = `${OUTPUT_DIR}/regional-sales.json`;
await df.toJson(jsonPath);
const jsonReloaded = await DataFrame.readJson(jsonPath);

const xlsxBytes = writeXlsx(df.columns, rows, { sheetName: "RegionalSales" });
const xlsxPath = `${OUTPUT_DIR}/regional-sales.xlsx`;
await writeFile(xlsxPath, xlsxBytes);
const xlsxReloaded = readXlsx(xlsxBytes);

const parquetBytes = writeParquet(df.columns, rows);
const parquetPath = `${OUTPUT_DIR}/regional-sales.parquet`;
await writeFile(parquetPath, parquetBytes);
const parquetReloaded = readParquet(parquetBytes);

console.log(`JSON rows reloaded:    ${jsonReloaded.shape[0]}`);
console.log(`XLSX rows reloaded:    ${xlsxReloaded.data.length}`);
console.log(`Parquet rows reloaded: ${parquetReloaded.data.length}`);

// ============================================================================
// Part 3: String accessor and styling
// ============================================================================
console.log("\nPart 3: String Accessor and Styling");
console.log("-".repeat(60));

const regionSeries = df.get("region");
console.log(`Upper-case regions:\n${regionSeries.str.upper().toString()}`);

// df.style builds a report by chaining formatting calls. The method names
// highlightMax, highlightMin and backgroundGradient follow pandas.
const styledHtml = df.style
  .setCaption("Daily Revenue Operations Snapshot")
  .highlightMax({ backgroundColor: "#dcfce7", fontWeight: "bold" })
  .backgroundGradient("#fff7ed", "#fb923c")
  .format("revenue", (value) => `$${Number(value).toLocaleString("en-US")}`)
  .format("supportTickets", (value) => `${value} tickets`)
  .toHTML();

const ansiTable = df.style
  .highlightMin({ backgroundColor: "#fee2e2" })
  .format("revenue", (value) => `$${Number(value).toLocaleString("en-US")}`)
  .toANSI();

const htmlPath = `${OUTPUT_DIR}/styled-sales-report.html`;
const ansiPath = `${OUTPUT_DIR}/styled-sales-report.txt`;
await writeFile(htmlPath, styledHtml, "utf-8");
await writeFile(ansiPath, ansiTable, "utf-8");

console.log(`Styled HTML report: ${htmlPath}`);
console.log(`Styled ANSI table:  ${ansiPath}`);

// ============================================================================
// Part 4: Plot accessor
// ============================================================================
console.log("\nPart 4: Plot Accessor");
console.log("-".repeat(60));

const trend = new DataFrame({
  day: [1, 2, 3, 4, 5, 6],
  revenue: rows.map((row) => row.revenue),
  supportTickets: rows.map((row) => row.supportTickets),
});

const trendFigure = trend.plot.line({
  x: "day",
  y: ["revenue", "supportTickets"],
  title: "Revenue vs Support Load",
  figsize: [800, 480],
});

const svgPath = `${OUTPUT_DIR}/revenue-support-trend.svg`;
await writeFile(svgPath, trendFigure.renderSVG().svg, "utf-8");

console.log(`Plot saved to: ${svgPath}`);

// ============================================================================
// Summary
// ============================================================================
console.log("\nKey Takeaways");
console.log("-".repeat(60));
console.log(
  "• JSON files are read and written by DataFrame. XLSX and Parquet have built-in codecs with no extra dependency."
);
console.log(
  "• toDatetime and dateRange build date columns, and the .dt and .str accessors clean them"
);
console.log("• df.style writes the same table as HTML or as an ANSI string for a terminal");
console.log("• df.plot.line() returns a Figure, and renderSVG() gives the SVG text");

console.log("\nDataFrame IO & Styling Example Complete!");
console.log("=".repeat(60));
