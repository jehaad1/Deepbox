# DataFrame Basics

> **View online:** https://deepbox.dev/examples/04-dataframe-basics

The basic DataFrame operations: creation, column access, selection, filtering, sorting, `head` and `tail`.

## Deepbox Modules Used

| Module              | Features Used                                                 |
| ------------------- | ------------------------------------------------------------- |
| `deepbox/dataframe` | `DataFrame`, `get`, `select`, `filter`, `sort`, `head`, `tail` |

## What It Shows

- A DataFrame is created from an object of equal-length column arrays.
- `get("age")` returns one column as a Series. `select(["name", "salary"])` returns a DataFrame with those columns.
- `filter((row) => ...)` keeps the rows for which the callback returns true.
- `sort("salary", false)` sorts descending. The second argument is `ascending`.
- `toArray()` converts a Series to a plain array.

## Usage

```bash
npm run example:04
```

## Output

Console output only.

## Files

```
04-dataframe-basics/
├── index.ts     # Main entry point
└── README.md    # This file
```
