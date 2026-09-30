# Catch the Rogue Bank: Detect-then-Aggregate Federated Intrusion Detection

**Subtitle:** Reputation-based poisoning detection plus class-balanced federated training on skewed NSL-KDD banks
**Team:** Athena Techne · **Tracks:** 🔴 Advanced (primary), 🟡 Intermediate (supporting)

## Results at a glance

All numbers are F1 on the notebook's held-out KDDTest+ split, from our fully-run Day 2 notebook.

| Track | Before (notebook baseline) | After (ours) | Gain |
|---|---|---|---|
| 🔴 **Advanced**: non-IID, bank 1 malicious (label-flip + ×15 scaling) | Naive FedAvg under attack: **0.1225** | **0.7758**, bank 1 detected and quarantined in round 1 | **+0.653** |
| 🟡 Intermediate: non-IID, no attack | Naive FedAvg: **0.7035** | **0.7629** | **+0.059** |

To separate the defense from the better model: with our *same* model and training, naive FedAvg under
the attack scores 0.3653, so the defense alone recovers **+0.41 F1**. Under attack, the defended model
(0.7758) matches or beats the no-attack model (0.7629): the attack is fully neutralized.

## 1. The problem we actually measured

The non-IID split is severe. Bank 4 is 96% normal traffic, bank 1 is 86% attacks (mostly probes), and bank 2
holds 43% of all data. Two things break:

- **Skew:** each bank's update mostly teaches "predict my majority class", and these pulls fight inside
  the average (0.764 IID → 0.704 non-IID).
- **Poisoning:** bank 1 flips its labels and multiplies its update by 15. One bank of five dominates the
  mean and F1 collapses to 0.12.

The two interact, and this became our central insight. Once bank 1 flips its labels, its update looks
like that of an honest *normal-heavy* bank such as bank 3 or 4. A naive "remove the odd one out" defense
either misses the attacker or throws out honest skewed banks. Our first defense did exactly that: it
quarantined honest banks 3 and 4 even with **no attack at all**. The rest of our work was about telling
"different because skewed" apart from "different because malicious".

## 2. Intermediate: fixing the skew (what worked, what didn't)

We changed one thing at a time on the notebook's split, then re-checked the winners over 3 random seeds.
F1 fluctuates by about ±0.02 between rounds, so single runs mislead.

| Change | F1 (3-seed mean) |
|---|---|
| Notebook baseline (WeakMLP, FedAvg) | 0.715 ± 0.006 |
| + **class-balanced local loss** (same model) | **0.745** ± 0.006 |
| + 64-32 MLP + **server momentum** (FedAvgM, β = 0.9) | **0.760** ± 0.006 |
| + signed-log input features | 0.712 (worse, dropped) |
| FedProx (μ = 0.01) | no measurable change (dropped) |

**Why the balanced loss works:** each bank weights positive examples by `n_neg / n_pos` of *its own*
labels. It needs no shared information, only local label counts, so it is privacy-preserving by
construction. Every bank's update then carries balanced "attack vs normal" information instead of
"my majority class", so updates stop cancelling. The ablation confirms it is the key ingredient: the
bigger MLP *without* balancing only reaches 0.723.

**Why server momentum helps:** skewed banks produce directions that swing from round to round, and
accumulating updates on the server smooths the oscillation.

**Why FedProx didn't help:** it limits client drift, but with one local epoch there is almost no drift to
limit. It is the wrong tool for this particular failure.

## 3. Advanced: detect, quarantine, aggregate

For every bank's update Δᵢ = localᵢ − global, each round:

1. **Magnitude test.** ‖Δᵢ‖ divided by the median norm. Scaling inflates it roughly 8–30×. Flag above **8×**.
2. **Direction test.** Cosine between Δᵢ and the coordinate-wise median update. A label-flipped bank pushes
   *against* the honest consensus. Flag below **0.15**. This catches the stealthy flip-only attack, which has no
   magnitude signal at all.
3. **Reputation.** Each flag is a strike; after 2 strikes the bank is **quarantined permanently**. From
   then on the median statistics are computed only from trusted banks, so an attacker cannot drag the
   reference toward itself.
4. **Robust aggregation of survivors.** Clip every update to 3× the median trusted norm, which bounds any single
   bank's influence without shrinking large honest banks. Then take the usual size-weighted mean, with server
   momentum.

The output is a **detection report** as well as a model: *"bank 1 flagged in round 1, quarantined from
round 2"*. A consortium can act on that by auditing the bank; a median defense cannot tell you who attacked.

### Calibrating without cheating

We set thresholds **only from a no-attack run** of the final training setup, recording what honest
banks do, the way an alarm threshold is set in production:

| Signal | Honest banks (worst case, 24 rounds) | Malicious bank |
|---|---|---|
| Norm ratio | ≤ 5.9× | 3.1× to 32× |
| Cosine to median update | ≥ +0.31 | median −0.02, down to −0.64 |

Thresholds (8× and 0.15) sit outside the honest range. Neither was tuned on attacked runs or on test F1.

**What went wrong first, and why it matters:** our thresholds were initially calibrated on the notebook's
WeakMLP (honest cosine went down to −0.22 there). When we switched to the balanced loss, honest bank 4's
updates grew to 5.9× the median, because balancing up-weights its rare attack rows. The old 4× rule then
quarantined an honest bank, and clipping every update to the median norm starved bank 2. F1 fell to 0.713.
Recalibrating on the new honest behaviour and clipping at 3× fixed both. **Lesson: a detector's thresholds
belong to a specific training setup, and any change to local training requires re-calibration.**

## 4. Does it generalize? Robustness sweep

Same model and training for every row; only the aggregation changes (seed 42, full table in the notebook).

| Scenario | Naive FedAvg | Median | **Ours** | Quarantined |
|---|---|---|---|---|
| No attack | 0.7629 | 0.7203 | **0.7619** | none |
| Scale ×15, bank 1 (default) | 0.3653 | 0.7517 | **0.7758** | [1] |
| Scale ×50, bank 1 | 0.0074 | 0.7505 | **0.7758** | [1] |
| Label-flip only, bank 1 | 0.7165 | 0.7422 | **0.7758** | [1] |
| Scale ×15, banks 1 + 3 | 0.7255 | 0.7064 | **0.7506** | [1, 3] |
| Label-flip only, banks 1 + 3 | 0.7192 | 0.7121 | **0.7535** | [1, 3] |

- **Zero false positives:** no honest bank was quarantined in the no-attack run on 5 of 5 seeds, and
  the defense costs essentially nothing when nobody attacks (0.7619 vs 0.7629).
- **Attack strength stops mattering:** ×15, ×50 and flip-only all end at exactly 0.7758. Once bank 1 is
  quarantined, training is identical to never having had the attacker.
- **Median is not enough:** it survives scaling but still mixes in part of the poisoned signal, and it
  loses 0.04 F1 even with no attack because it discards honest information from skewed banks.
- **Two colluding attackers (2 of 5 banks)** are both caught. Our first version missed the flip-only pair.
- **An honest oddity:** naive FedAvg only drops to 0.7255 with two scaled attackers. Bank 1 is attack-heavy
  and bank 3 normal-heavy, so their flipped, boosted updates largely cancel. That is luck, not safety.

## 5. Limitations and what we would do next

- **Adaptive attackers.** An attacker who knows our thresholds could scale by 5× and poison only
  slightly. The clipping bounds the damage, but detection would miss it. Next steps: track each bank's
  cosine *trend* across rounds, and use a small server-held validation set (FLTrust-style).
- **Majority assumption.** Median-based references fail once malicious banks are a majority (3 or more of 5).
- **Calibration data.** We assume a trusted no-attack warm-up exists. Without it, thresholds could be set from
  early rounds with conservative margins.
- **Test set.** KDDTest+ contains attack types absent from training, which caps recall around 0.65 for
  every method. Our gains come from aggregation and robustness, not from fitting that set.

## Reproducibility

The notebook runs top to bottom on CPU (about 15 minutes with the sweep). Seeds are fixed, and our cells restore
the random-number state, so the organizers' own cells reproduce the published baselines. The repo contains
`model_scripted.pt`, `submission.json`, all source code, and raw multi-seed results.
