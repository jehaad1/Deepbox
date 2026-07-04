# DataFrame IO & Styling

> **View online:** https://deepbox.dev/examples/47-dataframe-io-styling

A practical DataFrame operations example for the v1.0.0 polish layer: JSON/XLSX/Parquet round-trips, datetime parsing, styling, and pandas-like plotting accessors.

## Deepbox Modules Used

| Module               | Features Used                                                                 |
| -------------------- | ----------------------------------------------------------------------------- |
| `deepbox/dataframe`  | DataFrame, date_range, to_datetime, readXlsx, writeXlsx, readParquet, writeParquet, style, plot |

## Usage

```bash
npm run example:47
```

## Output

- JSON, XLSX, and Parquet example files in `output/`
- Styled HTML and ANSI reports
- SVG trend chart rendered with `df.plot.line()`

## Architecture

```text
47-dataframe-io-styling/
├── index.ts
├── README.md
└── output/
```
