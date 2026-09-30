"""Final system (mlp64 + balanced loss + server momentum) with and without the defense, under attack."""
import numpy as np
import torch

import harness as h
from defense import RobustAggregator
from experiments import SCENARIOS
from fl_ext import MLP, ServerMomentum, make_local_train, plain_fedavg, run_fl_v2


def median_agg(local_models, sizes, prev_global_model):
    return h.coordinate_median(local_models)


def run(scenario, agg, seed):
    torch.manual_seed(seed)
    det = None
    if agg == "fedavg":
        a = ServerMomentum(plain_fedavg, 0.9)
    elif agg == "median":
        a = ServerMomentum(median_agg, 0.9)
    else:
        det = RobustAggregator(h.get_flat_params, h.set_flat_params, verbose=False)
        a = ServerMomentum(det, 0.9)
    _, hist = run_fl_v2(h.noniid_clients, model_fn=MLP, local_train_fn=make_local_train(balanced=True),
                        agg_fn=a, rounds=8, verbose=False, **SCENARIOS[scenario])
    return hist[-1]["f1"], det


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]  # e.g. "ours" to run just our defense
    for sc in SCENARIOS:
        seeds = {"no attack": [42, 1, 2, 3, 4], "scale x15, bank 1 (notebook default)": [42, 1, 2]}.get(sc, [42])
        print(f"== {sc}  (seeds {seeds})", flush=True)
        for agg in (only or ["fedavg", "median", "ours"]):
            res = [run(sc, agg, s) for s in seeds]
            f1s = [r[0] for r in res]
            det = res[0][1]
            extra = (f"  quarantined per seed={[sorted(r[1].quarantined) for r in res]}"
                     f"  seed-{seeds[0]} flagged-by-round={[l['flagged'] for l in det.log]}" if det else "")
            print(f"   {agg:7s} F1 mean={np.mean(f1s):.4f} per-seed={f1s}{extra}", flush=True)
