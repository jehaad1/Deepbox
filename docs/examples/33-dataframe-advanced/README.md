# DataFrame Advanced Features

> **View online:** https://deepbox.dev/examples/33-dataframe-advanced

Twelve short parts on `Series` and `DataFrame` features: string and datetime accessors, rolling, expanding and EWM windows, `query` and `eval` expressions, `assign`, pivot tables, crosstabs, `nlargest`, interpolation and gap filling.

## Deepbox Modules Used

| Module              | Features Used                                                                                                                                      |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deepbox/dataframe` | Series.str, Series.dt, rolling, expanding, ewm, query, eval, assign, pivotTable, crosstab, nlargest, nsmallest, interpolate, ffill, bfill, fillna  |

## Usage

```bash
npm run example:33
```

## Output

- Console output only: each part prints its input and its result.
- `rolling` accepts `minPeriods` and `center`.
- `ewm` defaults to `adjust: false` and `bias: true`, which differs from pandas. Pass `{ adjust: true, bias: false }` to match pandas.
- The older names `pivot_table` and `dayofweek` still work. Use `pivotTable` and `dayOfWeek`.

## Files

```
33-dataframe-advanced/
├── index.ts     # Example script
└── README.md    # This file
```
