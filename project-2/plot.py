#!/usr/bin/env python3
"""Matplotlib fallback for plot.gp (same figure).  Also draws the retained-vs-L1 trade-off.
Usage: plot.py OUTDIR PREFIX"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

outdir, stem = sys.argv[1], sys.argv[2]
q = np.loadtxt(os.path.join(outdir, f"{stem}.query.hist"))
r = np.loadtxt(os.path.join(outdir, "reference.hist"))
s = np.loadtxt(os.path.join(outdir, f"{stem}.rescaled.hist"))
summ = dict(l.rstrip("\n").split("\t", 1) for l in open(os.path.join(outdir, f"{stem}.summary.tsv")))

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(q[:, 0], q[:, 1], color="#9400D3", lw=1.5, label=f"Query (n = {int(q[:, 2].sum()):,})")
ax.plot(r[:, 0], r[:, 1], color="#2E8B57", lw=1.5, label="Reference")
ax.plot(s[:, 0], s[:, 1], "*", color="#1E90FF", ms=4, label=f"Rescaled (n = {int(s[:, 2].sum()):,})")
ax.set_xlim(0, 700)
ax.set_ylim(bottom=0)
ax.set_xlabel("Fragment length (bp)")
ax.set_ylabel("Normalized frequency")
ax.set_title(f"Rescaled to reference: L1 = {summ['L1_rescaled_vs_reference']}, "
             f"retained {float(summ['retained_fraction']):.1%}")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(outdir, f"{stem}.rescaled.png"), dpi=130)

t = np.loadtxt(os.path.join(outdir, f"{stem}.tradeoff.tsv"), skiprows=1)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(t[:, 1] / 1e6, t[:, 2], color="#444")
ax.axvline(int(summ["retained"]) / 1e6, color="#1E90FF", ls="--", label=f"chosen: {summ['rule']}")
ax.set_xlabel("Retained fragments (millions)")
ax.set_ylabel("L1 distance to reference")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(outdir, f"{stem}.tradeoff.png"), dpi=130)
print(os.path.join(outdir, f"{stem}.rescaled.png"))
