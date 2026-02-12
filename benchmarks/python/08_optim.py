"""
Benchmark 08 — Optimizers & LR Schedulers
PyTorch
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import (
        StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
        LinearLR, OneCycleLR, ReduceLROnPlateau,
    )
except ImportError:
    print("⚠ PyTorch not installed. Run: pip3 install torch")
    sys.exit(0)

from utils import run, create_suite, header, footer

suite = create_suite("optim", "PyTorch")
header("Benchmark 08 — Optimizers & LR Schedulers", "PyTorch")

# ── Helper ──────────────────────────────────────────────

def make_model(in_f, hidden):
    return nn.Sequential(nn.Linear(in_f, hidden), nn.ReLU(), nn.Linear(hidden, 1))

def train_loop(optim_fn, epochs, batch, in_f):
    m = make_model(in_f, 32)
    opt = optim_fn(m.parameters())
    tX = torch.randn(batch, in_f)
    tY = torch.randn(batch, 1)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((m(tX) - tY) ** 2).mean()
        loss.backward()
        opt.step()

# ── Optimizer Creation ──────────────────────────────────

base_model = make_model(10, 32)

run(suite, "SGD create", "—", lambda: optim.SGD(base_model.parameters(), lr=0.01))
run(suite, "SGD create (momentum)", "—", lambda: optim.SGD(base_model.parameters(), lr=0.01, momentum=0.9))
run(suite, "Adam create", "—", lambda: optim.Adam(base_model.parameters(), lr=0.001))
run(suite, "AdamW create", "—", lambda: optim.AdamW(base_model.parameters(), lr=0.001))
run(suite, "Adagrad create", "—", lambda: optim.Adagrad(base_model.parameters(), lr=0.01))
run(suite, "AdaDelta create", "—", lambda: optim.Adadelta(base_model.parameters(), lr=1.0))
run(suite, "Nadam create", "—", lambda: optim.NAdam(base_model.parameters(), lr=0.002))
run(suite, "RMSprop create", "—", lambda: optim.RMSprop(base_model.parameters(), lr=0.01))

# ── Optimizer Step ──────────────────────────────────────

def make_step(opt_class, **kwargs):
    m = make_model(10, 32)
    opt = opt_class(m.parameters(), **kwargs)
    tX = torch.randn(16, 10)
    tY = torch.randn(16, 1)
    loss = ((m(tX) - tY) ** 2).mean()
    loss.backward()
    return lambda: opt.step()

run(suite, "SGD step", "16x10→1", make_step(optim.SGD, lr=0.01))
run(suite, "Adam step", "16x10→1", make_step(optim.Adam, lr=0.001))
run(suite, "AdamW step", "16x10→1", make_step(optim.AdamW, lr=0.001))
run(suite, "Adagrad step", "16x10→1", make_step(optim.Adagrad, lr=0.01))
run(suite, "RMSprop step", "16x10→1", make_step(optim.RMSprop, lr=0.01))

# ── Training Loops (per optimizer) ──────────────────────

run(suite, "SGD train 50 epochs", "32x10→1", lambda: train_loop(lambda p: optim.SGD(p, lr=0.01), 50, 32, 10), warmup=2, iterations=5)
run(suite, "SGD+momentum train 50 epochs", "32x10→1", lambda: train_loop(lambda p: optim.SGD(p, lr=0.01, momentum=0.9), 50, 32, 10), warmup=2, iterations=5)
run(suite, "Adam train 50 epochs", "32x10→1", lambda: train_loop(lambda p: optim.Adam(p, lr=0.01), 50, 32, 10), warmup=2, iterations=5)
run(suite, "AdamW train 50 epochs", "32x10→1", lambda: train_loop(lambda p: optim.AdamW(p, lr=0.01), 50, 32, 10), warmup=2, iterations=5)
run(suite, "Adagrad train 50 epochs", "32x10→1", lambda: train_loop(lambda p: optim.Adagrad(p, lr=0.01), 50, 32, 10), warmup=2, iterations=5)
run(suite, "AdaDelta train 50 epochs", "32x10→1", lambda: train_loop(lambda p: optim.Adadelta(p, lr=1.0), 50, 32, 10), warmup=2, iterations=5)
run(suite, "Nadam train 50 epochs", "32x10→1", lambda: train_loop(lambda p: optim.NAdam(p, lr=0.002), 50, 32, 10), warmup=2, iterations=5)
run(suite, "RMSprop train 50 epochs", "32x10→1", lambda: train_loop(lambda p: optim.RMSprop(p, lr=0.01), 50, 32, 10), warmup=2, iterations=5)

run(suite, "Adam train 100 epochs", "64x50→1", lambda: train_loop(lambda p: optim.Adam(p, lr=0.001), 100, 64, 50), warmup=1, iterations=3)
run(suite, "SGD train 100 epochs", "64x50→1", lambda: train_loop(lambda p: optim.SGD(p, lr=0.01), 100, 64, 50), warmup=1, iterations=3)

# ── LR Schedulers ───────────────────────────────────────

def sched_step(make_sched, steps):
    m = make_model(10, 16)
    opt = optim.SGD(m.parameters(), lr=0.1)
    sched = make_sched(opt)
    def fn():
        for _ in range(steps):
            sched.step()
    return fn

run(suite, "StepLR (100 steps)", "—", sched_step(lambda o: StepLR(o, step_size=10, gamma=0.1), 100))
run(suite, "MultiStepLR (100 steps)", "—", sched_step(lambda o: MultiStepLR(o, milestones=[30, 60, 80], gamma=0.1), 100))
run(suite, "ExponentialLR (100 steps)", "—", sched_step(lambda o: ExponentialLR(o, gamma=0.95), 100))
run(suite, "CosineAnnealingLR (100 steps)", "—", sched_step(lambda o: CosineAnnealingLR(o, T_max=100), 100))
run(suite, "LinearLR (100 steps)", "—", sched_step(lambda o: LinearLR(o, start_factor=0.1, total_iters=100), 100))
run(suite, "OneCycleLR (100 steps)", "—", sched_step(lambda o: OneCycleLR(o, max_lr=0.1, total_steps=100), 100))

def sched_step_plateau(steps):
    m = make_model(10, 16)
    opt = optim.SGD(m.parameters(), lr=0.1)
    sched = ReduceLROnPlateau(opt, patience=10)
    def fn():
        for _ in range(steps):
            sched.step(1.0)
    return fn

run(suite, "ReduceLROnPlateau (100 steps)", "—", sched_step_plateau(100))

# WarmupLR: PyTorch doesn't have a built-in WarmupLR; use LinearLR as equivalent
run(suite, "WarmupLR (100 steps)", "—", sched_step(lambda o: LinearLR(o, start_factor=0.01, total_iters=50), 100))

# ── State Dict ──────────────────────────────────────────

adam_opt = optim.Adam(base_model.parameters(), lr=0.001)
run(suite, "optimizer stateDict()", "Adam", lambda: adam_opt.state_dict())

sgd_opt = optim.SGD(base_model.parameters(), lr=0.01)
run(suite, "optimizer stateDict()", "SGD", lambda: sgd_opt.state_dict())

# ── zeroGrad ────────────────────────────────────────────

run(suite, "zeroGrad", "3-layer", lambda: adam_opt.zero_grad())

footer(suite, "pytorch-optim.json")
