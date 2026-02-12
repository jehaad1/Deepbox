"""
Benchmark 07 — Neural Networks
PyTorch
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("⚠ PyTorch not installed. Run: pip3 install torch")
    sys.exit(0)

from utils import run, create_suite, header, footer

suite = create_suite("nn", "PyTorch")
header("Benchmark 07 — Neural Networks", "PyTorch")

# ── Layer Creation ──────────────────────────────────────

run(suite, "Linear create", "10→64", lambda: nn.Linear(10, 64))
run(suite, "Linear create", "64→128", lambda: nn.Linear(64, 128))
run(suite, "Linear create", "128→256", lambda: nn.Linear(128, 256))
run(suite, "Sequential create", "10→64→1", lambda: nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1)))
run(suite, "Sequential create", "50→128→64→1", lambda: nn.Sequential(nn.Linear(50, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)))
run(suite, "Conv1d create", "1→16 k=3", lambda: nn.Conv1d(1, 16, 3))
run(suite, "Conv2d create", "1→16 k=3", lambda: nn.Conv2d(1, 16, 3))
run(suite, "RNN create", "10→32", lambda: nn.RNN(10, 32))
run(suite, "LSTM create", "10→32", lambda: nn.LSTM(10, 32))
run(suite, "GRU create", "10→32", lambda: nn.GRU(10, 32))
run(suite, "BatchNorm1d create", "64", lambda: nn.BatchNorm1d(64))
run(suite, "LayerNorm create", "[64]", lambda: nn.LayerNorm([64]))

# ── Forward Pass ────────────────────────────────────────

model1 = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
model2 = nn.Sequential(nn.Linear(50, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
model3 = nn.Sequential(nn.Linear(10, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

x32_10 = torch.randn(32, 10)
x128_10 = torch.randn(128, 10)
x64_50 = torch.randn(64, 50)

run(suite, "forward (10→64→1)", "batch=32", lambda: model1(x32_10))
run(suite, "forward (10→64→1)", "batch=128", lambda: model1(x128_10))
run(suite, "forward (50→128→64→1)", "batch=64", lambda: model2(x64_50))
run(suite, "forward (10→256→128→64→1)", "batch=32", lambda: model3(x32_10))

# ── Activation Layers ───────────────────────────────────

act_in = torch.randn(32, 64)

run(suite, "ReLU forward", "32x64", lambda: nn.ReLU()(act_in))
run(suite, "Sigmoid forward", "32x64", lambda: nn.Sigmoid()(act_in))
run(suite, "Tanh forward", "32x64", lambda: nn.Tanh()(act_in))
run(suite, "LeakyReLU forward", "32x64", lambda: nn.LeakyReLU()(act_in))
run(suite, "ELU forward", "32x64", lambda: nn.ELU()(act_in))
run(suite, "GELU forward", "32x64", lambda: nn.GELU()(act_in))
run(suite, "Swish forward", "32x64", lambda: nn.SiLU()(act_in))
run(suite, "Mish forward", "32x64", lambda: nn.Mish()(act_in))
run(suite, "Softmax forward", "32x64", lambda: nn.Softmax(dim=-1)(act_in))

# ── Forward + Backward ─────────────────────────────────

x32_10g = torch.randn(32, 10, requires_grad=True)
x64_50g = torch.randn(64, 50, requires_grad=True)

def fwd_bwd_small():
    out = model1(x32_10g)
    target = torch.randn(32, 1)
    loss = ((out - target) ** 2).mean()
    loss.backward()

def fwd_bwd_large():
    out = model2(x64_50g)
    target = torch.randn(64, 1)
    loss = ((out - target) ** 2).mean()
    loss.backward()

run(suite, "forward+backward (10→64→1)", "batch=32", fwd_bwd_small)
run(suite, "forward+backward (50→128→64→1)", "batch=64", fwd_bwd_large, iterations=10)

# ── Loss Functions ──────────────────────────────────────

pred32 = torch.randn(32, 1)
target32 = torch.randn(32, 1)
pred32_10 = torch.randn(32, 10)
target32_cls = torch.randint(0, 10, (32,))
pred32_bin = torch.randn(32, 1)
target32_bin = torch.randint(0, 2, (32, 1)).float()

run(suite, "mseLoss", "32x1", lambda: nn.MSELoss()(pred32, target32))
run(suite, "maeLoss", "32x1", lambda: nn.L1Loss()(pred32, target32))
run(suite, "rmseLoss", "32x1", lambda: torch.sqrt(nn.MSELoss()(pred32, target32)))
run(suite, "huberLoss", "32x1", lambda: nn.HuberLoss()(pred32, target32))
run(suite, "crossEntropyLoss", "32x10", lambda: nn.CrossEntropyLoss()(pred32_10, target32_cls))
run(suite, "binaryCrossEntropyLoss", "32x1", lambda: nn.BCEWithLogitsLoss()(pred32_bin, target32_bin))

# ── Training Loops ──────────────────────────────────────

def train_adam_50():
    m = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = optim.Adam(m.parameters(), lr=0.01)
    tX = torch.randn(32, 10)
    tY = torch.randn(32, 1)
    for _ in range(50):
        opt.zero_grad()
        loss = ((m(tX) - tY) ** 2).mean()
        loss.backward()
        opt.step()

def train_sgd_50():
    m = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = optim.SGD(m.parameters(), lr=0.01)
    tX = torch.randn(32, 10)
    tY = torch.randn(32, 1)
    for _ in range(50):
        opt.zero_grad()
        loss = ((m(tX) - tY) ** 2).mean()
        loss.backward()
        opt.step()

def train_adam_100():
    m = nn.Sequential(nn.Linear(50, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = optim.Adam(m.parameters(), lr=0.001)
    tX = torch.randn(64, 50)
    tY = torch.randn(64, 1)
    for _ in range(100):
        opt.zero_grad()
        loss = ((m(tX) - tY) ** 2).mean()
        loss.backward()
        opt.step()

run(suite, "train Adam 50 epochs", "32x10→1", train_adam_50, warmup=2, iterations=5)
run(suite, "train SGD 50 epochs", "32x10→1", train_sgd_50, warmup=2, iterations=5)
run(suite, "train Adam 100 epochs", "64x50→1", train_adam_100, warmup=1, iterations=3)

# ── Inference (noGrad) ──────────────────────────────────

infer_model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
infer_model.eval()
infer_x256 = torch.randn(256, 10)
infer_x32 = torch.randn(32, 10)

def infer256():
    with torch.no_grad():
        infer_model(infer_x256)

def infer32():
    with torch.no_grad():
        infer_model(infer_x32)

run(suite, "inference (noGrad)", "batch=256", infer256)
run(suite, "inference (noGrad)", "batch=32", infer32)

# ── Module Operations ───────────────────────────────────

run(suite, "parameters()", "3-layer", lambda: list(model1.parameters()))
run(suite, "stateDict()", "3-layer", lambda: model1.state_dict())
run(suite, "train/eval toggle", "—", lambda: (model1.train(), model1.eval()))

footer(suite, "pytorch-nn.json")
