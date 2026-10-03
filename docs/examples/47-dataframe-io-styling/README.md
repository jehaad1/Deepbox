# DataFrame IO & Styling

> **View online:** https://deepbox.dev/examples/47-dataframe-io-styling

Reads and writes a small sales table as CSV, JSON, XLSX and Parquet, parses dates with `toDatetime`, formats the table with `df.style` as HTML and ANSI text, and draws a line chart with `df.plot.line()`.

## Deepbox Modules Used

| Module              | Features Used                                                                                 |
| ------------------- | --------------------------------------------------------------------------------------------- |
| `deepbox/dataframe` | DataFrame, dateRange, toDatetime, readXlsx, writeXlsx, readParquet, writeParquet, style, plot |

## Usage

```bash
npm run example:47
```

## Output

- Console output: the CSV round-trip shape, parsed weekday numbers and the number of rows read back from each file format.
- Files in `output/`: `regional-sales.json`, `regional-sales.xlsx`, `regional-sales.parquet`, `styled-sales-report.html`, `styled-sales-report.txt` and `revenue-support-trend.svg`.
- The older names `date_range`, `to_datetime`, `dayofweek`, `highlight_max`, `highlight_min` and `background_gradient` still work. Use `dateRange`, `toDatetime`, `dayOfWeek`, `highlightMax`, `highlightMin` and `backgroundGradient`.

## Files

```text
47-dataframe-io-styling/
├── index.ts
├── README.md
└── output/
```
