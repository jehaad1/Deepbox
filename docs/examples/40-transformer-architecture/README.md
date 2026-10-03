# Transformer Architecture

> **View online:** https://deepbox.dev/examples/40-transformer-architecture

Builds the parts of a Transformer one at a time and prints the tensor shapes: `MultiheadAttention`, encoder and decoder layers and stacks, `PositionalEncoding`, and a small encoder-decoder pipeline from token ids to vocabulary logits with `Embedding`, `causalMask` and a `Linear` output projection.

## Deepbox Modules Used

| Module            | Features Used                                                                                                                                                   |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deepbox/nn`      | MultiheadAttention, TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder, PositionalEncoding, Embedding, Linear, causalMask |
| `deepbox/ndarray` | tensor, randn, noGrad                                                                                                                                           |

## Usage

```bash
npm run example:40
```

## Output

- Console output only: input and output shapes for every component, parameter counts of the encoder and decoder stacks, and the logits shape of the pipeline.
- The inputs are random, so no numbers are printed. The example shows how the pieces connect, not a trained model.
- The forward passes run inside `noGrad()` and return plain tensors. Outside `noGrad()` the same calls return `GradTensor` values that support `backward()`.

## Files

```
40-transformer-architecture/
├── index.ts     # Example script
└── README.md    # This file
```
