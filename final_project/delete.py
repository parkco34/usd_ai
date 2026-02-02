#!/usr/bin/env python
"""
smoker_comparison_plots.py
==========================
PURPOSE:  Visualize how smoking_years, alcohol_units_per_week, and
          exercise_hours_per_week relate to lung_cancer_risk — but ONLY
          for people who are already smokers (smoker == 1).

WHY FILTER FIRST?
    By isolating smokers, we control for the binary smoking variable and
    can ask a sharper question: "Among people who DO smoke, which
    continuous features differ between those diagnosed and those not?"

PLOT CHOICE — side-by-side box plots:
    Box plots are ideal here because our target (lung_cancer_risk) is
    categorical (0 vs 1), while our features are numeric.  Box plots
    let us compare medians, spread (IQR), and outliers at a glance —
    exactly the kind of visual that supports inferential thinking
    ("do these two groups look like they came from the same distribution?").

LEARNING NOTES:
    - fig, axes = plt.subplots(...)  creates a Figure and an array of Axes.
      Think of Figure as the canvas, Axes as individual plot panels.
    - enumerate(zip(...)) is a Pythonic pattern for looping over
      parallel lists while also getting an index counter.
    - We use ax.set_*() instead of plt.xlabel() because when you have
      subplots, you need to talk to each Axes object directly.
"""

import pandas as pd
import matplotlib.pyplot as plt

# ── 1. LOAD & FILTER ────────────────────────────────────────────────
#    Read the full dataset, then create a filtered view.
#    .copy() avoids the SettingWithCopyWarning — it gives us an
#    independent DataFrame rather than a slice of the original.
df = pd.read_csv("./data/lung_cancer.csv")
smokers = df[df["smoker"] == 1].copy()

print(f"Total records:  {len(df)}")
print(f"Smokers only:   {len(smokers)}")
print(f"Diagnosed (1):  {smokers['lung_cancer_risk'].sum()}")
print(f"Not diagnosed:  {(smokers['lung_cancer_risk'] == 0).sum()}")

# ── 2. DEFINE WHAT WE WANT TO COMPARE ──────────────────────────────
#    Keep feature names and readable labels in parallel lists.
#    This makes the loop below clean and easy to extend later —
#    just add another entry to each list.
features = ["smoking_years", "alcohol_units_per_week", "exercise_hours_per_week"]
labels   = ["Smoking Years", "Alcohol (units/week)", "Exercise (hours/week)"]

# ── 3. SPLIT BY DIAGNOSIS ──────────────────────────────────────────
#    We create two Series for each feature — one per diagnosis group.
#    This is the manual approach so you can see exactly what goes into
#    each box.  (Seaborn would do this automatically, but explicit is
#    better when you're learning.)
diagnosed     = smokers[smokers["lung_cancer_risk"] == 1]
not_diagnosed = smokers[smokers["lung_cancer_risk"] == 0]

# ── 4. BUILD THE FIGURE ────────────────────────────────────────────
#    1 row × 3 columns, sharing nothing on y-axes (each feature has
#    its own natural scale).  figsize is (width, height) in inches.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

for i, (feat, label) in enumerate(zip(features, labels)):
    ax = axes[i]  # grab the i-th subplot panel

    # Build the two data groups for this feature
    data_to_plot = [not_diagnosed[feat], diagnosed[feat]]

    # bp = ax.boxplot(...)  returns a dictionary of Line2D objects
    # so we can style them after the fact if we want.
    bp = ax.boxplot(
        data_to_plot,
        tick_labels=["No Cancer (0)", "Cancer (1)"],
        patch_artist=True,           # fill boxes with color
        widths=0.5,                  # box width
        medianprops=dict(color="black", linewidth=2),
    )

    # Color each box — index 0 is "No Cancer", index 1 is "Cancer"
    colors = ["#5DADE2", "#E74C3C"]  # blue-ish, red-ish
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Axis labels & title
    ax.set_title(f"{label}\n(Smokers Only)", fontsize=13, fontweight="bold")
    ax.set_ylabel(label, fontsize=11)
    ax.set_xlabel("Lung Cancer Diagnosis", fontsize=11)

    # Light grid on y-axis only — helps read values without clutter
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

# ── 5. OVERALL TITLE & LAYOUT ──────────────────────────────────────
fig.suptitle(
    "Feature Distributions by Lung Cancer Diagnosis (Smokers Only, n={})".format(len(smokers)),
    fontsize=15,
    fontweight="bold",
    y=1.02,  # nudge above the subplots so it doesn't overlap
)

plt.tight_layout()
plt.savefig("smoker_comparison_plots.png", dpi=150, bbox_inches="tight")
plt.show()

