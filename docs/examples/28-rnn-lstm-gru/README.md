# Recurrent Neural Network Layers

> **View online:** https://deepbox.dev/examples/28-rnn-lstm-gru

Runs the `RNN`, `LSTM` and `GRU` layers on a small batch of sequences and prints the output shapes. It also shows stacked layers, unbatched input and the weight count of each layer type.

## Deepbox Modules Used

| Module            | Features Used  |
| ----------------- | -------------- |
| `deepbox/ndarray` | tensor, noGrad |
| `deepbox/nn`      | RNN, LSTM, GRU |

## Usage

```bash
npm run example:28
```

## Output

- Console output only: output shapes for each layer type, and a comparison of the number of weights in RNN, LSTM and GRU layers of the same size.

The forward passes run inside `noGrad()`, so they return plain tensors. Outside `noGrad()` the layers return a `GradTensor` that tracks the weights.
