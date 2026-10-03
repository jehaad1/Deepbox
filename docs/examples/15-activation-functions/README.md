# Activation Functions

> **View online:** https://deepbox.dev/examples/15-activation-functions

Apply common activation functions to a range of inputs, compare their values at a few points and plot them.

## Deepbox Modules Used

| Module            | Features Used                                                                                                        |
| ----------------- | -------------------------------------------------------------------------------------------------------------------- |
| `deepbox/ndarray` | `relu`, `sigmoid`, `softmax`, `gelu`, `leakyRelu`, `elu`, `mish`, `swish`, `softplus`, `relu6`, `selu`, `celu`, `softsign`, `hardswish`, `linspace` |
| `deepbox/plot`    | `Figure`, `plot`, `legend`, `renderSVG`                                                                              |

## What It Shows

- Thirteen element-wise activations are listed with their formula, a typical use, and their values at `x = -2, -1, 0, 1, 2`. `softmax` works on a whole vector and is shown separately.
- Each activation is available as a function (`relu(x)`) and as a tensor method (`x.relu()`).
- `gelu` uses the tanh approximation by default. PyTorch's default is the exact form. Pass `{ approximate: "none" }` to get the exact one. The script prints the largest gap between the two, about `5e-4`.
- Five curves are drawn on one plot with a legend.

## Usage

```bash
npm run example:15
```

## Output

One SVG file is written to `output/`: `activations.svg`, a comparison of ReLU, sigmoid, GELU, Swish and ELU.

## Files

```
15-activation-functions/
├── index.ts     # Main entry point
├── README.md    # This file
└── output/      # Generated SVG chart
```
