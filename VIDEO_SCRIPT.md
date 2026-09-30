# Video script: "Catch the Rogue Bank" (Athena Techne), target 2:50

About 420 spoken words. **[SCREEN]** = what to show; the rest is what you say. Speak slightly slower than feels
natural, and keep the pauses at the arrows (→).

---

### 0:00–0:20 · Hook
**[SCREEN]** Title slide: "Catch the Rogue Bank — Athena Techne", then the F1-over-rounds chart.

> Five banks train one intrusion detector together, without ever sharing customer data. That's federated
> learning. → But what happens when one of those banks turns malicious? → With plain federated averaging,
> our detector's F1 fell from 0.70 to **0.12**. We brought it back to **0.78**, and we can tell you exactly
> which bank attacked.

### 0:20–0:50 · The problem
**[SCREEN]** Notebook cell with the per-bank label mix (bank 4 = 96% normal, bank 1 = 86% attacks).

> Two things go wrong. First, the banks see very different traffic: bank 4 is almost all normal, and bank 1
> is mostly attacks. Averaging their updates makes them fight each other. → Second, bank 1 flips its labels
> and multiplies its update fifteen times, so it drowns out everyone else. → And here's the tricky part:
> once bank 1 flips its labels, it *looks like* an honest normal-heavy bank. Our first defense got fooled.
> It kicked out honest banks even when nobody was attacking.

### 0:50–1:20 · Fix 1: skewed data (Intermediate)
**[SCREEN]** The intermediate results line: 0.7035 → 0.7629.

> So first we fixed the skew. Each bank balances its own loss using only its own label counts, so nothing
> extra is shared. We add momentum on the server to smooth the swings between rounds. → That took clean
> non-IID F1 from 0.70 to 0.76, and it held on three random seeds. → We also tried FedProx and log features.
> They didn't help, and we explain why in the writeup.

### 1:20–2:05 · Fix 2: the defense (Advanced)
**[SCREEN]** The `[defense] round 1: flagged=[1]` log lines, then the detection report.

> Then the defense. Every round we check each bank's update twice. → Is it too *big*? Honest banks never
> went above about six times the median, and the attacker is fifteen times. → Is it pointing the *wrong way*?
> Honest banks always agree with the majority; the attacker pushes against it. → Two strikes and a bank is
> quarantined for good. The survivors are clipped and averaged. → Crucially, we set these thresholds using
> only a run with **no attack**, just how honest banks behave. We never tuned them on the attack or on test
> F1.

### 2:05–2:40 · Results
**[SCREEN]** The chart, then the robustness table from the notebook.

> Here's the result. Red is plain averaging under attack: it collapses. Grey is the median defense we were
> given. Blue is ours: **0.776**, which is even better than the unattacked baseline, with bank 1 caught in
> round 1. → We also stress-tested it: stronger attacks, a stealthy attack with no scaling, and two attackers
> at once. We caught them every time, with **zero false alarms** when there was no attack.

### 2:40–2:55 · Close
**[SCREEN]** Limitations bullet from the writeup, then the repo link.

> It's not perfect. An attacker who knows our thresholds could stay just under them, and we explain how
> we'd handle that next. → Everything is reproducible: the notebook, the exported model, and the code are
> in our repo. Thanks for watching. We're Athena Techne.

---

## Recording tips
- **Screen + voice is enough.** Record the notebook and chart with OBS or Loom, and add your face in a corner if you like.
- **Practise twice with a timer.** If you run over 3:00, trim the Fix 1 section first; Advanced is your primary track.
- **Say the numbers clearly**: 0.12 → 0.78, "bank 1 caught in round 1", "zero false alarms". Those are what judges remember.
- **Upload to YouTube as Unlisted**, and check the link works in a private or incognito window before adding it to the writeup.
