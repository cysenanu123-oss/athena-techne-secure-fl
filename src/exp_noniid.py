"""Clean non-IID: what raises the FedAvg ceiling? One change at a time, all on the notebook's split."""
import sys

import torch

import harness as h
from fl_ext import MLP, ServerMomentum, make_local_train, plain_fedavg, run_fl_v2

CONFIGS = {
    "A weak, fedavg (notebook baseline)": dict(),
    "B weak + balanced loss": dict(local_train_fn=make_local_train(balanced=True)),
    "C mlp64 + balanced": dict(model_fn=MLP, local_train_fn=make_local_train(balanced=True)),
    "D mlp64 + balanced + fedprox 0.01": dict(model_fn=MLP, local_train_fn=make_local_train(balanced=True, prox_mu=0.01)),
    "E mlp64 + balanced + server momentum .9": dict(model_fn=MLP, local_train_fn=make_local_train(balanced=True),
                                                    agg_fn="momentum"),
    "F mlp64, fedavg (no balancing)": dict(model_fn=MLP),
}

if __name__ == "__main__":
    for name, cfg in CONFIGS.items():
        if sys.argv[1:] and name[0] not in sys.argv[1]:
            continue
        cfg = dict(cfg)
        if cfg.get("agg_fn") == "momentum":
            cfg["agg_fn"] = ServerMomentum(plain_fedavg, beta=0.9)
        torch.manual_seed(h.SEED)
        _, hist = run_fl_v2(h.noniid_clients, rounds=8, verbose=False, **cfg)
        f1s = [m["f1"] for m in hist]
        print(f"{name:42s} final F1={f1s[-1]:.4f}  P={hist[-1]['precision']:.3f} R={hist[-1]['recall']:.3f}  "
              f"by round={f1s}", flush=True)
