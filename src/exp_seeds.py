"""Multi-seed check of the best non-IID candidates (clean non-IID split)."""
import numpy as np
import torch

import harness as h
from fl_ext import MLP, LogMLP, ServerMomentum, make_local_train, plain_fedavg, run_fl_v2

CONFIGS = {
    "A weak fedavg (baseline)": lambda: dict(),
    "B weak + balanced": lambda: dict(local_train_fn=make_local_train(balanced=True)),
    "E mlp64 + balanced + momentum": lambda: dict(model_fn=MLP, local_train_fn=make_local_train(balanced=True),
                                                  agg_fn=ServerMomentum(plain_fedavg, 0.9)),
    "G logmlp64 + balanced + momentum": lambda: dict(model_fn=LogMLP, local_train_fn=make_local_train(balanced=True),
                                                     agg_fn=ServerMomentum(plain_fedavg, 0.9)),
}
for name, cfg in CONFIGS.items():
    finals, lasts3 = [], []
    for seed in [42, 1, 2]:
        torch.manual_seed(seed)
        _, hist = run_fl_v2(h.noniid_clients, rounds=8, verbose=False, **cfg())
        finals.append(hist[-1]["f1"]); lasts3.append(np.mean([m["f1"] for m in hist[-3:]]))
    print(f"{name:34s} final F1 mean={np.mean(finals):.4f} +/- {np.std(finals):.4f}  per-seed={finals}  "
          f"(mean of last 3 rounds={np.mean(lasts3):.4f})", flush=True)
