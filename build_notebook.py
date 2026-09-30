"""Build the final Day 2 notebook: the organizers' notebook with our code in the YOUR TURN cells."""
import copy
import json
import re

SRC = "02-intermediate-advanced-day-2.ipynb"
OUT = "athena_techne_day2.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


def strip_docstring_imports(path, drop_prefixes):
    """Take a module's code without its module docstring and project-local imports."""
    s = open(path).read()
    s = re.sub(r'^""".*?"""\n', "", s, flags=re.S)
    lines = [l for l in s.splitlines() if not any(l.startswith(p) for p in drop_prefixes)]
    return "\n".join(lines).strip("\n")


fl_ext = strip_docstring_imports("fl_ext.py", ["import copy", "import numpy", "import torch", "import harness"])
fl_ext = fl_ext.replace("h.N_FEATURES", "N_FEATURES").replace("h.WeakMLP", "WeakMLP").replace("h.local_train", "local_train")
fl_ext = fl_ext.replace("h.get_flat_params", "get_flat_params").replace("h.set_flat_params", "set_flat_params")
fl_ext = fl_ext.replace("h.fedavg", "fedavg").replace("h.df_to_tensors", "df_to_tensors").replace("h.evaluate", "evaluate")
fl_ext = re.sub(r"\n\nclass LogMLP\(MLP\):.*", "", fl_ext, flags=re.S)  # tried, did not help -> not in final notebook
defense = strip_docstring_imports("defense.py", ["import copy", "import numpy", "import torch"])

INTERMEDIATE_MD = """
### Our approach (Athena Techne): fix the skew where it hurts, keep FedAvg's efficiency

Two small, principled changes, each targeting a concrete non-IID failure we measured:

1. **Class-balanced local loss.** Bank 4 is 96% normal traffic, bank 1 is 86% attacks. Under plain
   BCE, each bank's update mostly teaches "predict my majority class", and those pulls fight in the
   average. Each bank now weights positives by `n_neg / n_pos` of **its own** labels (only local label
   counts are needed, nothing is shared), so every bank's update carries balanced information.
2. **Server momentum (FedAvgM, beta = 0.9).** Skewed banks produce update directions that swing from round
   to round; accumulating them on the server smooths those oscillations.

Plus a moderately larger MLP (64-32 hidden, dropout 0.1). The FL loop below is the notebook's `run_fl`
with one addition: a pluggable local-training function (needed for the balanced loss). Attack
simulation, evaluation and the aggregation hook are unchanged.

Tried and dropped: FedProx (mu = 0.01) had no measurable effect (1 local epoch -> little drift to correct);
a signed-log input layer lowered F1 on KDDTest+ (0.712 vs 0.760, 3 seeds).
"""

INTERMEDIATE_CODE = f"""
# 🔧 YOUR TURN — intermediate track (Athena Techne)
{fl_ext}


_rng_state = torch.get_rng_state()  # restored below so the notebook's own later cells are unaffected
torch.manual_seed(SEED)
custom_model, custom_history = run_fl_v2(
    noniid_clients, model_fn=MLP, rounds=8,
    local_train_fn=make_local_train(balanced=True),
    agg_fn=ServerMomentum(plain_fedavg, beta=0.9),
)
print("\\nNaive non-IID FedAvg F1 (notebook baseline):", noniid_history[-1]["f1"])
print("Our non-IID training + aggregation F1:      ", custom_history[-1]["f1"])
torch.set_rng_state(_rng_state)
"""

ADVANCED_MD = """
### Our defense (Athena Techne): detect, quarantine, then aggregate robustly

Per round, for every bank's update `delta_i = local_i - previous_global`:

1. **Magnitude test.** `||delta_i|| / median norm`. The scaling attack inflates the update ~15x; honest
   banks never exceeded 5.9x in clean runs -> flag above **8x**.
2. **Direction test.** Cosine of `delta_i` with the coordinate-wise median update. A label-flipped bank
   pushes against the honest majority; honest (even very skewed) banks never went below **+0.31** in
   clean runs -> flag below **0.15**. This also catches the *stealthy* flip-only attack with no scaling.
3. **Reputation + quarantine.** Each flag is a strike; 2 strikes -> the bank is excluded permanently.
   The server's reference statistics (median norm / median update) are then computed only from
   non-quarantined banks, so an attacker cannot drag the reference toward itself.
4. **Robust aggregation of survivors.** Clip each update to 3x the median trusted norm (bounds any single
   bank's influence without shrinking large honest banks), then the usual size-weighted mean, with the
   same server momentum as the intermediate track.

Thresholds were calibrated **only on a no-attack run** of this training setup (the honest behaviour),
never on attacked runs or on test F1 - the same way you would set an alarm threshold in production.
Output is not just a better model but a **detection report**: which bank, flagged from which round.
"""

ADVANCED_CODE = f"""
# 🔧 YOUR TURN — advanced track (Athena Techne)
{defense}


def median_agg(local_models, sizes, prev_global_model):
    return coordinate_median(local_models)


ATTACK = dict(malicious_clients=[1], attack="scale", scale_factor=15.0)  # the notebook's scenario

# Fair "before": naive FedAvg under the SAME attack with the SAME model/training as our defense
torch.manual_seed(SEED)
naive_same_model, naive_same_model_history = run_fl_v2(
    noniid_clients, model_fn=MLP, rounds=8, local_train_fn=make_local_train(balanced=True),
    agg_fn=ServerMomentum(plain_fedavg, beta=0.9), verbose=False, **ATTACK)

torch.manual_seed(SEED)
detector = RobustAggregator(get_flat_params, set_flat_params, verbose=True)
my_defense_model, my_defense_history = run_fl_v2(
    noniid_clients, model_fn=MLP, rounds=8, local_train_fn=make_local_train(balanced=True),
    agg_fn=ServerMomentum(detector, beta=0.9), **ATTACK)

print("\\nDetection report: quarantined banks =", sorted(detector.quarantined),
      "| first flagged in round", next((l["round"] for l in detector.log if l["flagged"]), None))
print("\\nNaive FedAvg under attack (notebook WeakMLP):  ", attacked_history[-1]["f1"])
print("Naive FedAvg under attack (our model/training): ", naive_same_model_history[-1]["f1"])
print("Median defense (notebook baseline):             ", defended_history[-1]["f1"])
print("Our defense:                                    ", my_defense_history[-1]["f1"])
"""

SWEEP_MD = """
### Robustness sweep: does the defense generalize beyond the one scenario it was shown?

Stronger scaling, a stealthy flip-only attack (no magnitude signal), two colluding attackers, and
**no attack at all** (false-positive check). Same model and training everywhere; only the
aggregation changes. Takes a few minutes on CPU - set `RUN_SWEEP = False` to skip.
"""

SWEEP_CODE = """
RUN_SWEEP = True

SCENARIOS = {
    "no attack": dict(malicious_clients=[]),
    "scale x15, bank 1 (notebook default)": dict(malicious_clients=[1], attack="scale", scale_factor=15.0),
    "scale x50, bank 1": dict(malicious_clients=[1], attack="scale", scale_factor=50.0),
    "label-flip only, bank 1 (stealthy)": dict(malicious_clients=[1], attack="label_flip"),
    "scale x15, banks 1+3": dict(malicious_clients=[1, 3], attack="scale", scale_factor=15.0),
    "label-flip only, banks 1+3": dict(malicious_clients=[1, 3], attack="label_flip"),
}

def sweep_run(scenario, agg):
    torch.manual_seed(SEED)
    det = None
    if agg == "fedavg":
        a = ServerMomentum(plain_fedavg, 0.9)
    elif agg == "median":
        a = ServerMomentum(median_agg, 0.9)
    else:
        det = RobustAggregator(get_flat_params, set_flat_params, verbose=False)
        a = ServerMomentum(det, 0.9)
    _, hist = run_fl_v2(noniid_clients, model_fn=MLP, local_train_fn=make_local_train(balanced=True),
                        agg_fn=a, rounds=8, verbose=False, **SCENARIOS[scenario])
    return hist[-1]["f1"], (sorted(det.quarantined) if det else None)

if RUN_SWEEP:
    rows = []
    for sc in SCENARIOS:
        r = {"scenario": sc}
        for agg in ["fedavg", "median", "ours"]:
            f1, q = sweep_run(sc, agg)
            r[agg] = f1
            if q is not None:
                r["quarantined"] = q
        rows.append(r)
        print(r, flush=True)
    sweep_df = pd.DataFrame(rows).set_index("scenario")
    display(sweep_df)
"""

PLOT_CODE = """
import matplotlib.pyplot as plt

rounds = range(1, 9)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(rounds, [m["f1"] for m in attacked_history], "o-", color="#c0392b", label="Naive FedAvg under attack")
ax.plot(rounds, [m["f1"] for m in defended_history], "s-", color="#7f8c8d", label="Coordinate-median defense")
ax.plot(rounds, [m["f1"] for m in noniid_history], "--", color="#95a5a6", label="Naive FedAvg, no attack (non-IID)")
ax.plot(rounds, [m["f1"] for m in my_defense_history], "D-", color="#1f6feb", lw=2.5, label="Our defense (under attack)")
ax.set_xlabel("Communication round"); ax.set_ylabel("F1 on held-out KDDTest+")
ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(loc="lower right")
ax.set_title("Bank 1 turns malicious (label-flip + 15x scaling), non-IID banks")
plt.tight_layout(); plt.savefig("f1_over_rounds.png", dpi=160); plt.show()
"""

nb = json.load(open(SRC))
cells = nb["cells"]
new = []
for i, c in enumerate(cells):
    c = copy.deepcopy(c)
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
    if i == 16:
        # keep the organizers' example cell unchanged so the notebook's later baseline cells consume the
        # random-number stream exactly as in the original notebook (reproducible "before" numbers)
        new += [c, md(INTERMEDIATE_MD), code(INTERMEDIATE_CODE)]
    elif i == 21:
        new += [md(ADVANCED_MD), code(ADVANCED_CODE), md(SWEEP_MD), code(SWEEP_CODE), code(PLOT_CODE)]
    elif i == 23:
        s = "".join(c["source"])
        s = s.replace('TEAM_NAME = "CHANGE_ME"', 'TEAM_NAME = "Athena Techne"')
        s = s.replace('TRACK = "intermediate"  # "intermediate" or "advanced"', 'TRACK = "advanced"  # "intermediate" or "advanced"')
        s = s.replace("FINAL_MODEL = custom_model", "FINAL_MODEL = my_defense_model")
        c["source"] = s
        new.append(c)
    else:
        new.append(c)
nb["cells"] = new
json.dump(nb, open(OUT, "w"), indent=1)
print(f"wrote {OUT} with {len(new)} cells")
