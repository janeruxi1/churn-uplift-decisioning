"""Draw the README's figures from the recorded results (no data needed).

    python scripts/make_readme_figure.py   ->  reports/figures/00_readme_hero.png      (break-even vs evidence)
                                               reports/figures/00_readme_summary.png   (three questions)
                                               reports/figures/00_readme_value.png     (value vs the unknown effect)
"""
import json
from pathlib import Path

# --- shared README figure style (same block in every repo) ---------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 12, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white",
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def save(fig, path):
    fig.tight_layout(w_pad=3)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print("wrote", path.name)
# ---------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "reports" / "metrics"


def rec(name):
    return json.loads((M / f"{name}.json").read_text(encoding="utf-8"))


model, camp, dec = rec("02_kkbox_churn_model"), rec("05_campaign_effect"), rec("07_retention_decision")
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1.25, 1.15]})

ax = axes[0]
curve = dec["reach_curve"]
x = [100 * c["share_contacted"] for c in curve]
y = [100 * c["churners_reached"] for c in curve]
ax.plot([0, 100], [0, 100], color=GREY, lw=1.5, ls=(0, (4, 3)))
ax.text(70, 58, "at random", color=MUTED, fontsize=10)
ax.plot(x, y, color=BLUE, lw=2.5)
cap = model["march"]["lightgbm"]["capture_10"]
ax.plot(10, 100 * cap, "o", ms=9, color=BLUE, mec="white", mew=2)
ax.annotate(f"top 10% of scores\nreach {cap:.0%} of churners", (10, 100 * cap), (26, 40), fontsize=10.5,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.set(xlim=(0, 100), ylim=(0, 102), xlabel="% of members contacted, highest score first",
       ylabel="% of March churners reached")
ax.set_title(f"1 · The churn model finds churners\n(PR-AUC {model['march']['lightgbm']['pr_auc']:.3f}, "
             f"{model['mar_churn']:.1%} churn)")

ax = axes[1]
rows = [("Called vs held back, raw", camp["effect_unadjusted"]), ("Propensity weighted", camp["effect_ipw"]),
        ("Outcome model (g-computation)", camp["effect_gcomp_20"]), ("Doubly robust (the estimate)", camp["effect_aipw"])]
for i, (label, e) in enumerate(rows[::-1]):
    c = BLUE if label.startswith("Doubly") else GREY
    ax.plot([100 * e["ci"][0], 100 * e["ci"][1]], [i, i], color=c, lw=2.5, solid_capstyle="round")
    ax.plot(100 * e["diff"], i, "o", ms=9, color=c, mec="white", mew=2)
    ax.text(100 * e["ci"][1] + 0.12, i, f"{100 * e['diff']:+.2f}".replace("-", "−"), va="center", fontsize=10.5)
ax.axvline(0, color=MUTED, lw=1)
ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
ax.set_xlim(-1.9, 1.2)
ax.set_xlabel("change in churn from the call (percentage points, 95% interval)")
ax.grid(axis="y", visible=False)
ax.set_title(f"2 · A real retention call: no detectable effect\n({camp['n']:,} customers; every interval spans zero)")

ax = axes[2]
pol = {p["effect"]: p for p in dec["policy"]}
def fewer(r):
    return "no effect or worse" if r <= 0 else f"{r:.0%} less churn"


labels = [f"{lab}:\n{fewer(pol[k]['relative_reduction'])}"
          for k, lab in (("low", "Low end"), ("point", "Best estimate"), ("high", "High end"))]
vals = [pol[k]["members_worth_offering"] for k in ("low", "point", "high")]
bars = ax.barh(labels[::-1], vals[::-1], color=[GREY, BLUE, GREY][::-1], height=0.55)
for bar, v in zip(bars, vals[::-1]):
    ax.text(v + 600, bar.get_y() + bar.get_height() / 2, f"{v:,}", va="center", fontsize=11)
ax.set_xlim(0, max(vals) * 1.25)
ax.set_xlabel("members a free month pays for, by the call's effect")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.grid(axis="y", visible=False)
ax.set_title(f"3 · Whether a free month pays is unknown:\ntest it first ({dec['test_size']['n_per_arm']:,} members per arm)")

save(fig, ROOT / "reports" / "figures" / "00_readme_summary.png")


r = dec
R, be = r["relative_reduction"], r["breakeven"]

# ---------- option 1: break-even against the evidence
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.axvspan(100 * R["low"], 100 * R["high"], color="#eaf2fc", lw=0)
ax.plot([100 * R["low"], 100 * R["high"]], [1, 1], color=BLUE, lw=3, solid_capstyle="round")
ax.plot(100 * R["point"], 1, "o", ms=11, color=BLUE, mec="white", mew=2)
ax.text(100 * R["high"] + 1.5, 1, f"best estimate {R['point']:.0%} fewer churners\n95% interval {R['low']:+.0%} to {R['high']:+.0%}"
        .replace("-", "−"), va="center", fontsize=10.5)
XMAX = 75
rows = [("most promising 1% of members", be["top_1"]), ("most promising 5%", be["top_5"]),
        ("most promising 10%", be["top_10"])]
for i, (lab, v) in enumerate(rows):
    y = -0.05 - 0.42 * i
    if 100 * v <= XMAX:
        ax.plot(100 * v, y, "D", ms=10, color=ORANGE, mec="white", mew=2)
        ax.text(100 * v + 1.5, y, f"{lab}: {v:.0%}", va="center", fontsize=10.5)
        ax.axvline(100 * v, color=ORANGE, lw=1, ls=(0, (3, 3)), ymin=0.05, ymax=0.95)
    else:
        ax.annotate("", (XMAX, y), (XMAX - 9, y), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2))
        ax.text(XMAX - 10, y, f"{lab}: {v:.0%}, off the chart", va="center", ha="right", fontsize=10.5)
ax.axvline(0, color=MUTED, lw=1)
ax.set_yticks([1, -0.47], ["What the evidence says\n(a real retention call)", "What a free month needs\nto pay for itself"])
ax.set_ylim(-1.2, 1.6)
ax.set_xlim(-25, XMAX)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%".replace("-", "−")))
ax.set_xlabel("relative cut in churn from the offer")
ax.grid(axis="y", visible=False)
ax.set_title(f"A free month pays only where it cuts churn by {be['top_1']:.0%} or more, and the evidence can't yet "
             f"say it does:\ntest it on the highest-risk members first ({r['test_size']['n_per_arm']:,} per arm)")
save(fig, ROOT / "reports" / "figures" / "00_readme_hero.png")

# ---------- option 2: value against the unknown effect
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.axvspan(100 * R["low"], 100 * R["high"], color="#eaf2fc", lw=0)
ax.text(100 * R["low"] + 1, 5.6, "95% interval of the\nmeasured effect", color=MUTED, fontsize=10, va="top")
for m, c, lw in ((0.2, GREY, 1.5), (0.3, BLUE, 2.8), (0.4, GREY, 1.5)):
    g = sorted((x for x in r["value_grid"] if abs(x["margin"] - m) < 1e-9), key=lambda x: x["relative_reduction"])
    xs = [100 * x["relative_reduction"] for x in g]
    ys = [x["net_value_ntd"] / 1e6 for x in g]
    ax.plot(xs, ys, color=c, lw=lw)
    ax.text(xs[-1] + 1, ys[-1], f"{m:.0%} margin", va="center", fontsize=10, color=INK if c == BLUE else MUTED)
pt = next(p for p in r["policy"] if p["effect"] == "point")
ax.plot(100 * R["point"], pt["net_value_ntd"] / 1e6, "o", ms=10, color=BLUE, mec="white", mew=2)
ax.annotate(f"best estimate: {pt['members_worth_offering']:,} members worth a free month,\n"
            f"NT${pt['net_value_ntd'] / 1e6:.2f}M a month", (100 * R["point"], pt["net_value_ntd"] / 1e6),
            (100 * R["point"] - 20, 2.6), fontsize=10.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.axvline(0, color=MUTED, lw=1)
ax.set_xlim(-20, 70)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%".replace("-", "−")))
ax.set_xlabel("relative cut in churn from a free month (assumed)")
ax.set_ylabel("net value, NT$ million a month")
ax.set_title("The free month's value hangs on an effect nobody has measured: anywhere from zero to\n"
             "NT$3M a month inside the evidence, so test before rolling out")
save(fig, ROOT / "reports" / "figures" / "00_readme_value.png")
