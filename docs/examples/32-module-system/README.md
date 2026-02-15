# Neural Network Module System

> **View online:** https://deepbox.dev/examples/32-module-system

Demonstrates the Module base class: custom modules, parameter registration, state serialization,
train/eval modes, freeze/unfreeze, and Sequential container.

## Deepbox Modules Used

| Module            | Features Used                                       |
| ----------------- | --------------------------------------------------- |
| `deepbox/ndarray` | tensor, parameter, GradTensor                       |
| `deepbox/nn`      | Module, Linear, ReLU, Sequential, stateDict, freeze |

## Usage

```bash
npm run example:32
```

## Output

- Console output showing module construction, parameter enumeration, state dict serialization,
  train/eval toggling, and freeze/unfreeze behavior
