#!/usr/bin/env python3
"""
tcga_boxplot.py -- Project 4: expression of a gene in Primary Tumor vs Solid Tissue Normal
from GDC/TCGA STAR "augmented_star_gene_counts" files.

Input
  --sample-sheet gdc_sample_sheet.tsv   (columns used: File ID, File Name, Sample Type [, Case ID, Sample ID])
  either --data-dir DIR                (extracted tcga_data.tar.gz: DIR/<File ID>/<File Name>)
  or     --tar tcga_data.tar.gz        (streamed directly, no extraction to disk)
Output (prefix):
  <prefix>.values.tsv   one row per file: ids, sample type, TPM, log2(TPM+1)
  <prefix>.stats.tsv    n, median, IQR per group; Mann-Whitney U and Welch t-test p-values
  <prefix>.png/.pdf     notched boxplot, Y = log2(TPM + 1), n per group, p-value
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

GROUPS = ["Solid Tissue Normal", "Primary Tumor"]
TPM_COL = "tpm_unstranded"  # 7th column of the STAR counts file


def load_sheet(path: str, groups: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    need = {"File ID", "File Name", "Sample Type"}
    if not need <= set(df.columns):
        sys.exit(f"sample sheet lacks columns {need - set(df.columns)}")

    def norm(st: str) -> str | None:
        # merged entries look like "Primary Tumor, Primary Tumor"; keep only unambiguous ones
        parts = {p.strip() for p in str(st).split(",")}
        return parts.pop() if len(parts) == 1 else None

    # the GDC sheet lists every data type (VCFs, SNP arrays, ...); keep the STAR gene-count files
    rna = df["File Name"].str.endswith("augmented_star_gene_counts.tsv", na=False)
    if "Data Type" in df:
        rna |= df["Data Type"].eq("Gene Expression Quantification") & df["File Name"].str.endswith(".tsv", na=False)
    df = df[rna].copy()
    df["group"] = df["Sample Type"].map(norm)
    return df[df["group"].isin(groups)].reset_index(drop=True)


def extract_value(data: bytes, gene: bytes, col: int) -> float:
    """Value of column `col` (0-based) on the row whose gene_name (2nd column) == gene,
    or whose gene_id (1st column, version stripped) == gene."""
    key = b"\t" + gene + b"\t"
    i = data.find(key)
    if i < 0:  # maybe an Ensembl id was given
        j = data.find(b"\n" + gene)
        if j < 0:
            return float("nan")
        i = j + 1
    s = data.rfind(b"\n", 0, i) + 1
    e = data.find(b"\n", i)
    fields = data[s : e if e >= 0 else None].split(b"\t")
    return float(fields[col])


def tpm_column(header_blob: bytes) -> int:
    for line in header_blob.split(b"\n"):
        if line.startswith(b"gene_id"):
            cols = line.rstrip(b"\r").split(b"\t")
            return cols.index(TPM_COL.encode())
    return 6  # documented position (7th column)


def values_from_dir(sheet: pd.DataFrame, data_dir: str, gene: str, threads: int) -> list[float]:
    def one(row) -> float:
        p = os.path.join(data_dir, row["File ID"], row["File Name"])
        if not os.path.exists(p):
            p2 = os.path.join(data_dir, row["File Name"])
            if not os.path.exists(p2):
                return float("nan")
            p = p2
        with open(p, "rb") as fh:
            data = fh.read()
        return extract_value(data, gene.encode(), tpm_column(data[:4096]))

    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(one, [r for _, r in sheet.iterrows()]))


def values_from_tar(sheet: pd.DataFrame, tar_path: str, gene: str) -> list[float]:
    want = {f"{r['File ID']}/{r['File Name']}": i for i, r in sheet.iterrows()}
    by_name = {r["File Name"]: i for i, r in sheet.iterrows()}
    out = [float("nan")] * len(sheet)
    try:
        import rapidgzip

        raw = rapidgzip.open(tar_path, parallelization=os.cpu_count() or 1)
    except ImportError:
        import gzip

        raw = gzip.open(tar_path, "rb")
    with raw, tarfile.open(fileobj=raw, mode="r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name.lstrip("./")
            idx = want.get(name, by_name.get(os.path.basename(name)))
            if idx is None:
                continue
            data = tf.extractfile(m).read()
            out[idx] = extract_value(data, gene.encode(), tpm_column(data[:4096]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tumor vs normal expression boxplot for one gene (TCGA STAR counts).")
    ap.add_argument("-s", "--sample-sheet", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("-d", "--data-dir", help="extracted tcga_data directory")
    src.add_argument("-t", "--tar", help="tcga_data.tar.gz (read directly)")
    ap.add_argument("-g", "--gene", default="NKX2-1")
    ap.add_argument("--title", default="Lung Adenocarcinoma")
    ap.add_argument("-o", "--out-prefix", help="default: results/<gene>")
    ap.add_argument("-j", "--threads", type=int, default=16)
    a = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    prefix = a.out_prefix or os.path.join("results", a.gene)
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    sheet = load_sheet(a.sample_sheet, GROUPS)
    vals = values_from_dir(sheet, a.data_dir, a.gene, a.threads) if a.data_dir else values_from_tar(sheet, a.tar, a.gene)
    sheet["tpm"] = vals
    missing = int(sheet["tpm"].isna().sum())
    if missing:
        sys.stderr.write(f"warning: {missing} files missing or without {a.gene}; excluded\n")
    sheet = sheet.dropna(subset=["tpm"])
    sheet["log2_tpm_plus1"] = np.log2(sheet["tpm"] + 1.0)
    keep = [c for c in ["File ID", "File Name", "Case ID", "Sample ID", "group", "tpm", "log2_tpm_plus1"] if c in sheet]
    sheet[keep].to_csv(f"{prefix}.values.tsv", sep="\t", index=False)

    g = {k: sheet.loc[sheet["group"] == k, "log2_tpm_plus1"].to_numpy() for k in GROUPS}
    if min(len(v) for v in g.values()) == 0:
        sys.exit(f"a group is empty: { {k: len(v) for k, v in g.items()} }")
    mw = stats.mannwhitneyu(g[GROUPS[1]], g[GROUPS[0]], alternative="two-sided")
    wt = stats.ttest_ind(g[GROUPS[1]], g[GROUPS[0]], equal_var=False)
    rows = []
    for k in GROUPS:
        v = g[k]
        rows.append({"group": k, "n": len(v), "median_log2_tpm1": np.median(v),
                     "q1": np.percentile(v, 25), "q3": np.percentile(v, 75), "mean": v.mean()})
    st = pd.DataFrame(rows)
    st["mannwhitney_p"] = mw.pvalue
    st["welch_t_p"] = wt.pvalue
    st.to_csv(f"{prefix}.stats.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.boxplot([g[k] for k in GROUPS], notch=True, widths=0.55, tick_labels=GROUPS,
               medianprops=dict(color="black"), flierprops=dict(marker="o", markersize=3, markerfacecolor="#333"))
    for i, k in enumerate(GROUPS, 1):
        ax.text(i - 0.36, np.median(g[k]), f"n = {len(g[k])}", rotation=90, va="center", ha="center", fontsize=9)
    top = max(v.max() for v in g.values())
    ax.text(1.5, top + 0.75, f"Mann-Whitney p = {mw.pvalue:.2g}", ha="center")
    ax.text(1.5, top + 0.3, f"Welch t-test p = {wt.pvalue:.2g}", ha="center", fontsize=9, color="#555")
    ax.set_ylim(min(0, min(v.min() for v in g.values())), top + 1.5)
    ax.set_ylabel("Log2 (TPM + 1)")
    ax.set_title(f"{a.title}\n{a.gene}")
    fig.tight_layout()
    fig.savefig(f"{prefix}.png", dpi=150)
    fig.savefig(f"{prefix}.pdf")
    print(st.to_string(index=False))
    print(f"files: {len(sheet)} used, {missing} missing;  outputs: {prefix}.png / .values.tsv / .stats.tsv")


if __name__ == "__main__":
    main()
