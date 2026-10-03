# CNN Layers

> **View online:** https://deepbox.dev/examples/27-cnn-layers

Runs the convolution and pooling layers `Conv1d`, `Conv2d`, `MaxPool2d` and `AvgPool2d` on small inputs and prints the output shapes. It also shows `padding: "same"` and `padding: "valid"`, and chains layers with `Sequential`.

## Deepbox Modules Used

| Module            | Features Used                                    |
| ----------------- | ------------------------------------------------ |
| `deepbox/ndarray` | `tensor`, `noGrad`                               |
| `deepbox/nn`      | Conv1d, Conv2d, MaxPool2d, AvgPool2d, Sequential |

## Usage

```bash
npm run example:27
```

## Notes

- A layer with trainable weights returns a `GradTensor` when gradient tracking is on. The forward passes here are inference only, so they run inside `noGrad()` and return plain tensors. Both kinds of tensor have `.shape` and `.toString()`, so no type check is needed to read them.
- Pooling layers have no parameters and return a plain tensor for plain input.

## Output

- Console output only: input and output shapes for each layer, the pooled values, and the number of parameter tensors in a small `Sequential` CNN.
