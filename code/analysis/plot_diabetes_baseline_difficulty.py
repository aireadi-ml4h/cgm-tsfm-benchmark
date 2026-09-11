"""Figure for report.md Sec 27, check 1: baseline zero-shot MAE by diabetes
status group, both models, all 5 horizons. Reads the existing check1 CSV,
produces no new numbers. Follows this project's existing figure convention
(matplotlib Agg backend, dpi=150, bbox_inches="tight",
code/preprocessing/qc/raw_stream_qc/report.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = "/path/to/aireadi-collab"
df = pd.read_csv(f"{REPO_ROOT}/chronos2_zeroshot_results/diabetes_status_check1_baseline_difficulty.csv")

groups = ["mean_healthy", "mean_pre_diabetes_lifestyle_controlled",
          "mean_oral_medication_and_or_non_insulin_injectable_medication_controlled",
          "mean_insulin_dependent"]
labels = ["healthy", "pre-diabetes", "oral medication", "insulin dependent"]
horizons = ["30min", "1h", "2h", "3h", "4h"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, model in zip(axes, ["chronos2", "timesfm"]):
    sub = df[df["model"] == model].set_index("horizon").loc[horizons]
    x = range(len(horizons))
    width = 0.2
    for i, (g, lab) in enumerate(zip(groups, labels)):
        ax.bar([xi + i * width for xi in x], sub[g], width=width, label=lab)
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(horizons)
    ax.set_title(model)
    ax.set_xlabel("horizon")
ax = axes[0]
ax.set_ylabel("MAE (mg/dL)")
axes[1].legend(fontsize=8, loc="upper left")
fig.suptitle("Zero-shot glucose-only MAE by diabetes status (report.md Sec 27, check 1)")
fig.tight_layout()
out_path = f"{REPO_ROOT}/zeroshot_results_reports/figures/diabetes_status_baseline_difficulty.png"
import os
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
