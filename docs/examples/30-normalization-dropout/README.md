# Normalization & Dropout Layers

> **View online:** https://deepbox.dev/examples/30-normalization-dropout

Runs `BatchNorm1d`, `LayerNorm` and `Dropout` on small inputs. `BatchNorm1d` and `Dropout` behave differently in train and eval mode, and the example shows both.

## Deepbox Modules Used

| Module            | Features Used                   |
| ----------------- | ------------------------------- |
| `deepbox/ndarray` | `tensor`, `noGrad`              |
| `deepbox/nn`      | BatchNorm1d, LayerNorm, Dropout |

## Usage

```bash
npm run example:30
```

## Output

- Console output only: output shapes, the per-feature mean after `BatchNorm1d`, the per-sample mean after `LayerNorm`, and dropout in train and eval mode.
- Dropout is random, so the zeroed positions differ from run to run. Surviving values are scaled by `1 / (1 - p)`.

`eval()` does not turn gradient tracking off. For inference, run the forward pass inside `noGrad()` to get a plain tensor.
