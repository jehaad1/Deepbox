# Attention & Transformer Layers

> **View online:** https://deepbox.dev/examples/29-attention-transformer

Runs `MultiheadAttention` and `TransformerEncoderLayer` on a short sequence. It shows self-attention, the attention weights (`needWeights`), a key padding mask, a causal mask, and the `activation` and `normFirst` options of the encoder layer.

## Deepbox Modules Used

| Module            | Features Used                                           |
| ----------------- | ------------------------------------------------------- |
| `deepbox/ndarray` | `tensor`, `noGrad`                                      |
| `deepbox/nn`      | MultiheadAttention, TransformerEncoderLayer, causalMask |

## Usage

```bash
npm run example:29
```

## Output

- Console output only: output shapes, attention weight shapes and sums, the weight a padding mask removes, and parameter tensor counts.
