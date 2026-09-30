"""Reputation-based robust aggregation for the Day 2 advanced track.

Plugs into the notebook hook: run_fl(..., aggregation="custom", agg_fn=RobustAggregator()).
Needs the notebook's get_flat_params / set_flat_params (passed in, so this file works both
imported and pasted into the notebook).

Per round, for every client update  delta_i = local_i - previous_global:
  1. Magnitude test: ||delta_i|| / median norm. A boosted ("scaled") update is an order of
     magnitude larger than honest ones -> flag above `norm_ratio_thresh`.
  2. Direction test: cosine of delta_i with the coordinate-wise median update. A label-flipped
     update pushes against the honest majority -> flag below `cos_thresh`.
     Both thresholds are calibrated on a NO-attack run of the same training setup (honest
     behaviour only), never on attacked runs or test F1. Defaults are for the final setup
     (MLP + class-balanced loss + server momentum): honest max ratio 5.9x, honest min cos 0.31.
  3. Reputation: a client flagged in a round gets a strike; once it has `quarantine_after`
     strikes it is excluded permanently (detection output = which bank, from which round).
  4. Aggregation of survivors: clip each update to `clip_mult` x the median trusted norm (bounds
     any single bank's influence without shrinking large honest banks), then the usual
     size-weighted FedAvg mean.
"""
import copy

import numpy as np
import torch


class RobustAggregator:
    def __init__(self, get_flat_params, set_flat_params, norm_ratio_thresh=8.0, cos_thresh=0.15,
                 clip_mult=3.0, quarantine_after=2, verbose=True):
        self.get_flat, self.set_flat = get_flat_params, set_flat_params
        self.norm_ratio_thresh, self.cos_thresh = norm_ratio_thresh, cos_thresh
        self.clip_mult, self.quarantine_after = clip_mult, quarantine_after
        self.verbose = verbose
        self.strikes, self.quarantined, self.log, self.round = {}, set(), [], 0

    def __call__(self, local_models, sizes, prev_global_model):
        self.round += 1
        g = self.get_flat(prev_global_model)
        deltas = torch.stack([self.get_flat(m) - g for m in local_models])
        n = len(local_models)

        # reference statistics come only from clients that are not already quarantined
        trusted = [i for i in range(n) if i not in self.quarantined] or list(range(n))
        norms = deltas.norm(dim=1)
        ratio = norms / norms[trusted].median().clamp_min(1e-12)
        ref = deltas[trusted].median(dim=0).values
        cos = torch.nn.functional.cosine_similarity(deltas, ref.unsqueeze(0), dim=1)

        flagged = [i for i in range(n) if ratio[i] > self.norm_ratio_thresh or cos[i] < self.cos_thresh]
        for i in flagged:
            self.strikes[i] = self.strikes.get(i, 0) + 1
            if self.strikes[i] >= self.quarantine_after:
                self.quarantined.add(i)
        keep = [i for i in range(n) if i not in flagged and i not in self.quarantined]
        if not keep:  # never aggregate nothing: fall back to the least suspicious client
            keep = [int(torch.argmax(cos))]

        clip = self.clip_mult * norms[trusted].median()
        agg_deltas = torch.stack([deltas[i] * min(1.0, float(clip / norms[i].clamp_min(1e-12))) for i in keep])
        w = torch.tensor([sizes[i] for i in keep], dtype=torch.float32)
        new = g + (agg_deltas * (w / w.sum()).unsqueeze(1)).sum(0)

        self.log.append({"round": self.round, "flagged": flagged, "quarantined": sorted(self.quarantined),
                         "kept": keep, "norm_ratio": [round(float(v), 2) for v in ratio],
                         "cos": [round(float(v), 2) for v in cos]})
        if self.verbose:
            print(f"   [defense] round {self.round}: flagged={flagged} quarantined={sorted(self.quarantined)} "
                  f"kept={keep} cos={[round(float(v), 2) for v in cos]}")
        out = copy.deepcopy(local_models[0])
        self.set_flat(out, new)
        return out
