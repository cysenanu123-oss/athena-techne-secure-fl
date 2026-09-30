# Athena Techne — Secure Federated Intrusion Detection (CAIRLab Secure AI Hackathon, Days 2–3)

**Primary track: 🔴 Advanced** (detect/resist a poisoning bank) · **Supporting: 🟡 Intermediate** (non-IID aggregation)

Five banks train a shared intrusion detector on NSL-KDD without pooling data. The data is skewed
across banks (non-IID), and bank 1 turns malicious (label-flip + 15× update scaling). Our system
**detects and quarantines the malicious bank in round 1** and recovers F1 fully, with no false
alarms when nobody is attacking.

## Headline results (F1 on the notebook's held-out KDDTest+ split)

| | Naive FedAvg (notebook) | Coordinate median (notebook) | **Ours** |
|---|---|---|---|
| 🔴 Non-IID + malicious bank 1 (scale ×15) | 0.1225 | 0.6865 | **0.7758** (bank 1 quarantined, round 1) |
| 🟡 Non-IID, no attack | 0.7035 | – | **0.7629** |

- **Advanced: F1 recovered = 0.7758 − 0.1225 = +0.653.** With the *same* stronger model, naive FedAvg under
  the attack scores 0.3653, so the defense itself recovers +0.41 on top of the model improvement.
- **Intermediate: +0.059** over naive FedAvg on the same non-IID split.

### Robustness sweep (same model/training everywhere, seed 42)

| Scenario | Naive FedAvg | Median | **Ours** | Quarantined |
|---|---|---|---|---|
| No attack | 0.7629 | 0.7203 | 0.7619 | none (no false alarms) |
| Scale ×15, bank 1 (notebook default) | 0.3653 | 0.7517 | **0.7758** | [1] |
| Scale ×50, bank 1 | 0.0074 | 0.7505 | **0.7758** | [1] |
| Label-flip only, bank 1 (stealthy, no scaling) | 0.7165 | 0.7422 | **0.7758** | [1] |
| Scale ×15, banks 1+3 | 0.7255 | 0.7064 | **0.7506** | [1, 3] |
| Label-flip only, banks 1+3 | 0.7192 | 0.7121 | **0.7535** | [1, 3] |

Multi-seed checks (`results/`): no-attack false positives = 0 on 5/5 seeds; default attack 0.760 mean over
3 seeds with bank 1 quarantined on 3/3 (same-model naive FedAvg: 0.289).

## How it works

**Intermediate (non-IID):** class-balanced local loss (each bank weights positives by its *own*
`n_neg/n_pos`), server momentum (FedAvgM, β=0.9), and a 64-32 MLP.

**Advanced (defense), per round, per bank update Δᵢ = localᵢ − global:**
1. Magnitude test: ‖Δᵢ‖ / median norm > 8 → flag (honest max observed: 5.9×).
2. Direction test: cos(Δᵢ, coordinate-median Δ) < 0.15 → flag (honest min observed: 0.31).
3. Reputation: 2 flags → permanent quarantine; reference statistics then use trusted banks only.
4. Aggregate survivors: clip to 3× median trusted norm, size-weighted mean, server momentum.

Thresholds are calibrated **only on a no-attack run** (honest behaviour), never on attacked runs or test F1.

## Repository layout

| Path | What |
|---|---|
| `athena_techne_day2.ipynb` | Final Day 2 notebook, fully run (organizers' notebook + our code in the 🔧 cells) |
| `model_scripted.pt`, `submission.json` | Exported model + metrics, produced by the notebook's submission cell |
| `f1_over_rounds.png` | F1 per round: naive FedAvg under attack vs median vs our defense |
| `src/harness.py` | Verbatim copy of the notebook's data/partition/model/FL harness (sections 0–4) |
| `src/fl_ext.py` | Balanced local loss, FedProx option, server momentum, `run_fl_v2` |
| `src/defense.py` | `RobustAggregator`: detection + quarantine + robust aggregation |
| `src/experiments.py`, `exp_noniid.py`, `exp_seeds.py`, `exp_final.py` | Scenario sweeps and multi-seed checks |
| `results/` | Raw outputs of the sweeps |
| `WRITEUP.md` | Project writeup |

## Reproduce

**Notebook (recommended):** open `athena_techne_day2.ipynb` in Kaggle or Colab and *Run All* (CPU, ~15 min
including the robustness sweep; set `RUN_SWEEP = False` to skip it, ~3 min). It downloads NSL-KDD itself and
writes `model_scripted.pt`, `submission.json` and `f1_over_rounds.png`.

**Scripts:**
```bash
pip install torch scikit-learn pandas numpy matplotlib
cd src
python experiments.py            # WeakMLP: FedAvg / median / trimmed mean / ours across 6 scenarios
python exp_seeds.py              # non-IID training variants over 3 seeds
python exp_final.py              # final system: FedAvg / median / ours, multi-seed
```

**Verify the exported model:**
```python
import torch
m = torch.jit.load("model_scripted.pt")   # input: 41 standardized features in FEATURE_COLS order
```
