/**
 * Example 47: DataFrame IO & Styling
 *
 * Demonstrates v1.0.0 DataFrame operational polish: JSON/XLSX/Parquet round-trips,
 * string/date helpers, style rendering, and pandas-like plotting accessors.
 */

import { mkdir, writeFile } from "node:fs/promises";
import {
  DataFrame,
  date_range,
  readParquet,
  readXlsx,
  to_datetime,
  writeParquet,
  writeXlsx,
} from "deepbox/dataframe";

const OUTPUT_DIR = "docs/examples/47-dataframe-io-styling/output";

console.log("=".repeat(60));
console.log("Example 47: DataFrame IO & Styling");
console.log("=".repeat(60));

await mkdir(OUTPUT_DIR, { recursive: true });

const orderDates = date_range("2026-03-01", 6, "D");
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
// Part 1: CSV ingest + datetime helpers
// ============================================================================
console.log("\n📥 Part 1: CSV + Date Helpers");
console.log("-".repeat(60));

const csvRoundTrip = DataFrame.fromCsvString(df.toCsvString());
const parsedDates = to_datetime(rows.map((row) => row.orderDate));

console.log(`Round-trip CSV shape: ${csvRoundTrip.shape[0]} rows × ${csvRoundTrip.shape[1]} cols`);
console.log(
  `Parsed weekday numbers: ${Array.from(parsedDates.dt.dayofweek().data, (value) => String(value)).join(", ")}`
);

// ============================================================================
// Part 2: JSON, XLSX, and Parquet round-trips
// ============================================================================
console.log("\n🗃️  Part 2: JSON / XLSX / Parquet");
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
// Part 3: String accessor + styling
// ============================================================================
console.log("\n🎨 Part 3: String Accessor + Styling");
console.log("-".repeat(60));

const regionSeries = df.get("region");
console.log(`Upper-case regions: ${regionSeries.str.upper().toString()}`);

const styledHtml = df.style
  .setCaption("Daily Revenue Operations Snapshot")
  .highlight_max({ backgroundColor: "#dcfce7", fontWeight: "bold" })
  .background_gradient("#fff7ed", "#fb923c")
  .format("revenue", (value) => `$${Number(value).toLocaleString("en-US")}`)
  .format("supportTickets", (value) => `${value} tickets`)
  .toHTML();

const ansiTable = df.style
  .highlight_min({ backgroundColor: "#fee2e2" })
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
console.log("\n📈 Part 4: Plot Accessor");
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
console.log("\n💡 Key Takeaways");
console.log("-".repeat(60));
console.log("• DataFrame supports JSON file IO directly and zero-dependency XLSX/Parquet codecs");
console.log("• Date helpers and string accessors make operational reports easier to clean");
console.log("• df.style can emit HTML or ANSI-ready reports for notebooks and terminals");
console.log("• df.plot offers pandas-like plotting without leaving the Deepbox stack");

console.log("\n✅ DataFrame IO & Styling Example Complete!");
console.log("=".repeat(60));
