# Normalization & Dropout Layers

> **View online:** https://deepbox.dev/examples/30-normalization-dropout

Demonstrates BatchNorm1d, LayerNorm, and Dropout for training stability and regularization.

## Deepbox Modules Used

| Module            | Features Used                   |
| ----------------- | ------------------------------- |
| `deepbox/ndarray` | tensor, GradTensor              |
| `deepbox/nn`      | BatchNorm1d, LayerNorm, Dropout |

## Usage

```bash
npm run example:30
```

## Output

- Console output demonstrating normalization behavior in train vs eval modes
- Dropout masking and inverted scaling during training
