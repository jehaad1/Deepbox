# DataFrame GroupBy & Aggregation

> **View online:** https://deepbox.dev/examples/05-dataframe-groupby

Group rows by one or more columns and aggregate each group, like SQL `GROUP BY`.

## Deepbox Modules Used

| Module              | Features Used                                       |
| ------------------- | --------------------------------------------------- |
| `deepbox/dataframe` | `DataFrame`, `groupBy`, `agg`, `getGroup`, `nunique` |

## What It Shows

- `groupBy("product").agg({ quantity: "sum", revenue: "sum" })` applies one function per column.
- Named aggregation: `agg({ unitsSold: ["quantity", "sum"] })` chooses the output column name.
- `groupBy(["product", "region"])` groups on two columns.
- `getGroup("Laptop")` returns the rows of one group. `nunique()` counts distinct values per group.
- Groups appear in order of first occurrence. Pass `{ sort: true }` to `groupBy` for sorted keys.

## Usage

```bash
npm run example:05
```

## Output

Console output only.

## Files

```
05-dataframe-groupby/
├── index.ts     # Main entry point
└── README.md    # This file
```
