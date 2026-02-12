# Learning Rate Schedulers

Control the learning rate during training for better convergence. Deepbox provides 8 schedulers inspired by PyTorch.

## Deepbox Modules Used

| Module          | Features Used                                                                                                  |
| --------------- | -------------------------------------------------------------------------------------------------------------- |
| `deepbox/nn`    | Linear, ReLU, Sequential                                                                                       |
| `deepbox/optim` | Adam, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, LinearLR, ReduceLROnPlateau, WarmupLR, OneCycleLR |

## Usage

```bash
npm run example:16
```

## Output

- Console output showing learning rate progression for all 8 scheduler types
