"""Compare aggregation rules across attack scenarios on the notebook's non-IID split (WeakMLP, 8 rounds)."""
import copy
import sys

import numpy as np
import torch

import harness as h
from defense import RobustAggregator


def trimmed_mean(k=1):
    def agg(local_models, sizes, prev_global_model):
        flats = torch.stack([h.get_flat_params(m) for m in local_models]).sort(dim=0).values
        out = copy.deepcopy(local_models[0])
        h.set_flat_params(out, flats[k:len(local_models) - k].mean(0))
        return out
    return agg


SCENARIOS = {
    "no attack": dict(malicious_clients=[]),
    "scale x15, bank 1 (notebook default)": dict(malicious_clients=[1], attack="scale", scale_factor=15.0),
    "scale x50, bank 1": dict(malicious_clients=[1], attack="scale", scale_factor=50.0),
    "label-flip only, bank 1 (stealthy)": dict(malicious_clients=[1], attack="label_flip"),
    "scale x15, banks 1+3": dict(malicious_clients=[1, 3], attack="scale", scale_factor=15.0),
    "label-flip only, banks 1+3": dict(malicious_clients=[1, 3], attack="label_flip"),
}


def run(scenario, agg):
    torch.manual_seed(h.SEED)
    kw = dict(SCENARIOS[scenario], rounds=8, verbose=False)
    if agg == "ours":
        d = RobustAggregator(h.get_flat_params, h.set_flat_params, verbose=False)
        _, hist = h.run_fl(h.noniid_clients, aggregation="custom", agg_fn=d, **kw)
        det = f" flagged-by-round={[l['flagged'] for l in d.log]} quarantined={sorted(d.quarantined)}"
    elif agg == "trimmed":
        _, hist = h.run_fl(h.noniid_clients, aggregation="custom", agg_fn=trimmed_mean(1), **kw); det = ""
    else:
        _, hist = h.run_fl(h.noniid_clients, aggregation=agg, **kw); det = ""
    return hist[-1]["f1"], det


if __name__ == "__main__":
    aggs = sys.argv[1:] or ["fedavg", "median", "trimmed", "ours"]
    for sc in SCENARIOS:
        print(f"== {sc}", flush=True)
        for a in aggs:
            f1, det = run(sc, a)
            print(f"   {a:8s} F1={f1:.4f}{det}", flush=True)
