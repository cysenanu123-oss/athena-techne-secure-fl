"""Extensions to the notebook's FL harness for the non-IID track.

run_fl_v2 is the notebook's run_fl with one addition: a pluggable local-training function, so
banks can use a class-balanced loss and/or a FedProx proximal term. Attack simulation, evaluation
and the aggregation hook are unchanged (same code paths as the notebook).
"""
import copy

import numpy as np
import torch
import torch.nn as nn

import harness as h


class MLP(nn.Module):
    def __init__(self, n_features=h.N_FEATURES, hidden=64, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_local_train(balanced=False, prox_mu=0.0, epochs=1, lr=0.05, batch_size=256):
    """balanced: weight positives by n_neg/n_pos of THIS bank's labels (each bank only needs its
    own label counts). prox_mu > 0 adds FedProx's (mu/2)*||w - w_global||^2 penalty."""
    def local_train(model, X, y):
        global_params = [p.detach().clone() for p in model.parameters()]
        model = copy.deepcopy(model)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        pos = float(y.sum())
        pw = torch.tensor((len(y) - pos) / max(pos, 1.0)) if balanced else None
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        n = len(X)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = loss_fn(model(X[idx]), y[idx])
                if prox_mu > 0:
                    loss = loss + prox_mu / 2 * sum(((p - g) ** 2).sum()
                                                    for p, g in zip(model.parameters(), global_params))
                loss.backward()
                opt.step()
        return model
    return local_train


class ServerMomentum:
    """FedAvgM wrapper around any aggregator: global <- global + v, v <- beta*v + aggregated delta."""
    def __init__(self, inner_agg, beta=0.9):
        self.inner, self.beta, self.v = inner_agg, beta, None

    def __call__(self, local_models, sizes, prev_global_model):
        g = h.get_flat_params(prev_global_model)
        delta = h.get_flat_params(self.inner(local_models, sizes, prev_global_model)) - g
        self.v = delta if self.v is None else self.beta * self.v + delta
        out = copy.deepcopy(prev_global_model)
        h.set_flat_params(out, g + self.v)
        return out


def plain_fedavg(local_models, sizes, prev_global_model):
    return h.fedavg(local_models, sizes)


def run_fl_v2(client_dfs, model_fn=lambda: h.WeakMLP(), rounds=8, local_train_fn=None,
              agg_fn=plain_fedavg, malicious_clients=None, attack="scale", scale_factor=15.0, verbose=True):
    """Same loop as the notebook's run_fl (aggregation="custom"), plus local_train_fn."""
    local_train_fn = local_train_fn or (lambda m, X, y: h.local_train(m, X, y))
    malicious_clients = malicious_clients or []
    global_model = model_fn()
    client_tensors = [h.df_to_tensors(df) for df in client_dfs]
    history = []
    for r in range(rounds):
        local_models, sizes = [], []
        global_flat = h.get_flat_params(global_model)
        for cid, (X, y) in enumerate(client_tensors):
            is_malicious = cid in malicious_clients
            if is_malicious:
                y = 1 - y  # label-flip (identical to the notebook)
            lm = local_train_fn(global_model, X, y)
            if is_malicious and attack == "scale":
                delta = h.get_flat_params(lm) - global_flat
                h.set_flat_params(lm, global_flat + scale_factor * delta)
            local_models.append(lm)
            sizes.append(len(X))
        global_model = agg_fn(local_models, sizes, global_model)
        metrics = h.evaluate(global_model)
        history.append(metrics)
        if verbose:
            print(f"round {r + 1:>2}: {metrics}")
    return global_model, history


class LogMLP(MLP):
    """MLP with a signed-log input layer: tames heavy-tailed standardized features (src_bytes etc.)
    inside the model, so the exported model still takes the raw 41-feature tensor."""
    def forward(self, x):
        return self.net(torch.sign(x) * torch.log1p(torch.abs(x))).squeeze(-1)
