# Preprocessing: Encoders

> **View online:** https://deepbox.dev/examples/17-preprocessing-encoders

Turn categorical and label data into numbers and back again with five encoders.

## Deepbox Modules Used

| Module               | Features Used                                                                    |
| -------------------- | -------------------------------------------------------------------------------- |
| `deepbox/ndarray`    | `tensor`, `reshape`                                                              |
| `deepbox/preprocess` | `LabelEncoder`, `OneHotEncoder`, `OrdinalEncoder`, `LabelBinarizer`, `MultiLabelBinarizer` |

## What It Shows

- `LabelEncoder` maps one list of labels to integers. Classes are sorted alphabetically, and `inverseTransform` restores the strings.
- `OneHotEncoder` and `OrdinalEncoder` expect a 2D input with one column per feature, so 1D values are reshaped to `[n, 1]` with `t.reshape([n, 1])`.
- `OrdinalEncoder` numbers categories alphabetically by default. Pass `categories: [["low", "medium", "high"]]` to fix the order.
- `OneHotEncoder` throws on an unseen category. `handleUnknown: "ignore"` encodes it as all zeros.
- `LabelBinarizer` handles single-label targets with more than two classes. `MultiLabelBinarizer` handles rows that carry several labels.
- Encoded output is `float64`.

## Usage

```bash
npm run example:17
```

## Output

Console output only, with encode and decode round trips.

## Files

```
17-preprocessing-encoders/
├── index.ts     # Main entry point
└── README.md    # This file
```
