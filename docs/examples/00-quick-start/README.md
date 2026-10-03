# Quick Start

> **View online:** https://deepbox.dev/examples/00-quick-start

A short tour of tensors, DataFrames and a first machine learning model. The script is about 50 lines.

## Deepbox Modules Used

| Module               | Features Used                                  |
| -------------------- | ---------------------------------------------- |
| `deepbox/ndarray`    | `tensor`, chained `add`, `mul`, `mean`, `item` |
| `deepbox/dataframe`  | `DataFrame` creation and printing              |
| `deepbox/ml`         | `LinearRegression` (`fit`, `predict`)          |
| `deepbox/preprocess` | `trainTestSplit`                               |

## What It Shows

- Tensor methods chain: `a.add(b).mul(2)` is the same as `mul(add(a, b), 2)`.
- `item()` turns a one-element tensor into a plain JavaScript number.
- A `DataFrame` is built from an object of equal-length columns.
- A linear regression is fitted on `y = 2x + 1` and predicts the held-out rows.

## Usage

```bash
npm run example:00
```

## Output

Console output only: tensor arithmetic and a mean, a printed DataFrame, and the model's predictions next to the true values.

## Files

```
00-quick-start/
├── index.ts     # Main entry point
└── README.md    # This file
```
