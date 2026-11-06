#!/usr/bin/env python3
"""
rescale.py -- Project 2: rescale a query BED file to a reference fragment-length distribution
by sampling lines (exact-count stratified sampling without replacement).

Model
-----
q(l)  : number of query fragments of length l = end - start
r(l)  : reference normalized frequency (sum 1)
Keeping k(l) <= q(l) lines of length l gives the distribution k(l)/sum(k).
  * Exact match needs k(l) = M r(l) for all l  =>  M <= M0 = min_l q(l)/r(l)  (tiny when the
    query is sparse somewhere, e.g. 300-400 bp).
  * We take k_M(l) = min(q(l), M r(l)): bins where the query is too sparse are kept entirely
    (the "points that miss out"), all others match the reference exactly.
    Retained R(M) = sum k_M and error E(M) = L1(k_M/R(M), r) both increase with M.
  * M is chosen by (default) the knee of the retained-vs-error curve, or by --tolerance (largest
    M with E <= tol) or --target-n (R(M) closest to n).
Sampling: for every length l exactly k(l) of the q(l) lines are chosen uniformly at random
without replacement (seeded) -> the output histogram equals k(l) exactly; output keeps input order.

Two streaming passes over the input, O(N) time, O(block + N/8) memory.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from numba import njit, prange
import numba as nb

from fastio import available_memory, gzip_writer, stream_blocks


# ------------------------------------------------------------------ Numba kernels
@njit(cache=True, nogil=True)
def _line_len(buf, p, end, LM):
    """Parse one line starting at p.  Returns (next_p, length_bin); bin -1 = not a BED record,
    bin LM = length outside [0, LM)."""
    q = p
    while buf[q] != 9 and buf[q] != 10:
        q += 1
    if buf[q] == 10:
        return q + 1, -1
    q += 1
    s = 0
    nd = 0
    while buf[q] >= 48 and buf[q] <= 57:
        s = s * 10 + (buf[q] - 48)
        q += 1
        nd += 1
    ok = nd > 0 and buf[q] == 9
    e = 0
    if ok:
        q += 1
        nd = 0
        while buf[q] >= 48 and buf[q] <= 57:
            e = e * 10 + (buf[q] - 48)
            q += 1
            nd += 1
        c = buf[q]
        ok = nd > 0 and (c == 9 or c == 10 or c == 13)
    while buf[q] != 10:
        q += 1
    if not ok:
        return q + 1, -1
    ln = e - s
    if ln < 0 or ln >= LM:
        return q + 1, LM
    return q + 1, ln


@njit(cache=True)
def chunk_bounds(buf, z, T):
    b = np.empty(T + 1, np.int64)
    b[0] = 0
    for t in range(1, T):
        p = (z * t) // T
        if p < b[t - 1]:
            p = b[t - 1]
        while p > 0 and p < z and buf[p - 1] != 10:
            p += 1
        b[t] = p
    b[T] = z
    return b


@njit(cache=True, parallel=True, nogil=True)
def length_hist(buf, bounds, LM):
    T = bounds.size - 1
    h = np.zeros((T, LM + 1), np.int64)
    skipped = np.zeros(T, np.int64)
    for t in prange(T):
        p = bounds[t]
        end = bounds[t + 1]
        while p < end:
            p, l = _line_len(buf, p, end, LM)
            if l < 0:
                skipped[t] += 1
            else:
                h[t, l] += 1
    return h, skipped


@njit(cache=True, parallel=True, nogil=True)
def select_lines(buf, bounds, cbase, off, sel, LM):
    """Keep the lines whose (length, global ordinal within that length) is selected.
    cbase[t, l] = global ordinal of the first length-l line of chunk t.  Stable output."""
    T = bounds.size - 1
    kb = np.zeros(T + 1, np.int64)
    for t in prange(T):  # pass A: kept bytes per chunk
        ordn = cbase[t].copy()
        p = bounds[t]
        end = bounds[t + 1]
        acc = 0
        while p < end:
            a = p
            p, l = _line_len(buf, p, end, LM)
            if l >= 0 and l < LM:
                if sel[off[l] + ordn[l]]:
                    acc += p - a
                ordn[l] += 1
        kb[t + 1] = acc
    for t in range(T):
        kb[t + 1] += kb[t]
    out = np.empty(kb[T], np.uint8)
    for t in prange(T):  # pass B: copy kept lines
        ordn = cbase[t].copy()
        p = bounds[t]
        end = bounds[t + 1]
        o = kb[t]
        while p < end:
            a = p
            p, l = _line_len(buf, p, end, LM)
            if l >= 0 and l < LM:
                if sel[off[l] + ordn[l]]:
                    out[o : o + (p - a)] = buf[a:p]
                    o += p - a
                ordn[l] += 1
    return out


# ------------------------------------------------------------------ model
def load_reference(path: str):
    d = np.loadtxt(path, ndmin=2)
    L = int(d[:, 0].max()) + 1
    r = np.zeros(L)
    np.add.at(r, d[:, 0].astype(np.int64), d[:, 1])
    if r.sum() <= 0:
        sys.exit("reference histogram is empty")
    return r / r.sum()


def tradeoff(q, r, npts=4000):
    """Retained R(M) and L1 error E(M) on a geometric grid of M in [M0, Mfull]."""
    m = r > 0
    ratio = q[m] / r[m]
    M0 = max(ratio.min(), 1e-9)
    Mfull = max(ratio.max(), M0 * 1.0001)
    Ms = np.geomspace(M0, Mfull, npts)
    K = np.minimum(q[None, :], Ms[:, None] * r[None, :])
    R = K.sum(1)
    E = np.abs(K / R[:, None] - r[None, :]).sum(1)
    return Ms, R, E


def choose_M(q, r, tolerance=None, target_n=None):
    Ms, R, E = tradeoff(q, r)
    if target_n is not None:
        i = int(np.argmin(np.abs(R - target_n)))
        how = f"target n={target_n}"
    elif tolerance is not None:
        ok = np.flatnonzero(E <= tolerance)
        i = int(ok[-1]) if ok.size else 0
        how = f"largest M with L1 <= {tolerance}"
    else:  # knee (Kneedle): max distance between normalised retained gain and error growth
        x = (R - R[0]) / max(R[-1] - R[0], 1e-12)
        y = (E - E[0]) / max(E[-1] - E[0], 1e-12)
        i = int(np.argmax(x - y))
        how = "knee of retained-vs-L1 curve"
    return Ms[i], how, (Ms, R, E)


def draw_selection(q, k, seed):
    """Flat boolean selection: for each length l, exactly k[l] of its q[l] ordinals are True."""
    off = np.zeros(q.size + 1, np.int64)
    np.cumsum(q, out=off[1:])
    sel = np.zeros(int(off[-1]) + 1, np.bool_)
    rng = np.random.default_rng(seed)
    for l in np.flatnonzero(k):
        n, kk = int(q[l]), int(k[l])
        if kk >= n:
            sel[off[l] : off[l] + n] = True
        else:
            sel[off[l] + rng.choice(n, size=kk, replace=False)] = True
    return off, sel


def write_hist(path, counts, lo, hi):
    """length <TAB> normalized frequency [<TAB> count, for integer histograms]."""
    tot = counts.sum()
    is_int = np.issubdtype(counts.dtype, np.integer)
    with open(path, "w") as fh:
        for l in range(lo, hi + 1):
            f = counts[l] / tot if tot else 0.0
            fh.write(f"{l}\t{f:.10g}\t{int(counts[l])}\n" if is_int else f"{l}\t{f:.10g}\n")


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Rescale a BED file to a reference fragment-length distribution.")
    ap.add_argument("-q", "--query", required=True, help="query BED(.gz)")
    ap.add_argument("-r", "--reference", required=True, help="reference histogram: length<TAB>normalized freq")
    ap.add_argument("-o", "--outdir", default="rescaled_output")
    ap.add_argument("-p", "--prefix", help="output prefix (default: query file name stem)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--tolerance", type=float, help="keep the most reads with L1(rescaled, reference) <= tol")
    g.add_argument("--target-n", type=int, help="keep (about) this many reads")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plain", action="store_true", help="write plain .bed instead of .bed.gz")
    ap.add_argument("-j", "--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("-m", "--mem", type=float, default=None, help="memory budget in GB (default 50%% available)")
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    nb.set_num_threads(max(1, min(a.threads, nb.config.NUMBA_NUM_THREADS)))
    budget = int(a.mem * (1 << 30)) if a.mem else available_memory() // 2
    block = int(min(256 << 20, max(32 << 20, budget // 4)))  # 256 MB blocks: same speed, less RAM
    low_mem = budget < (2 << 30)
    os.makedirs(a.outdir, exist_ok=True)
    stem = a.prefix or os.path.basename(a.query).split(".bed")[0]
    out_bed = os.path.join(a.outdir, f"{stem}.rescaled.bed" + ("" if a.plain else ".gz"))

    r = load_reference(a.reference)
    LM = r.size  # lengths >= LM have r = 0 and are never kept
    T = 4 * a.threads

    # pass 1: query length histogram
    q = np.zeros(LM + 1, np.int64)
    skipped = 0
    for buf, z in stream_blocks([a.query], block, min(a.threads, 16), low_mem):
        h, sk = length_hist(buf, chunk_bounds(buf, z, T), LM)
        q += h.sum(0)
        skipped += int(sk.sum())
    t1 = time.perf_counter()
    qL = q[:LM].astype(np.float64)
    N = int(q.sum())

    # choose the scale and the exact per-length counts
    M, how, (Ms, Rs, Es) = choose_M(qL, r, a.tolerance, a.target_n)
    k = np.floor(np.minimum(qL, M * r) + 1e-9).astype(np.int64)
    off, sel = draw_selection(q[:LM], k, a.seed)

    # pass 2: emit the selected lines (input order preserved)
    base = np.zeros(LM, np.int64)
    fh = open(out_bed, "wb") if a.plain else gzip_writer(out_bed)
    with fh:
        for buf, z in stream_blocks([a.query], block, min(a.threads, 16), low_mem):
            bounds = chunk_bounds(buf, z, T)
            h, _ = length_hist(buf, bounds, LM)
            h = h[:, :LM]
            cbase = base[None, :] + np.cumsum(h, axis=0) - h
            fh.write(memoryview(select_lines(buf, bounds, cbase, off, sel, LM)))
            base += h.sum(0)
    t2 = time.perf_counter()

    # reports
    lo = int(np.flatnonzero(r)[0])
    hi = LM - 1
    write_hist(os.path.join(a.outdir, f"{stem}.query.hist"), q[:LM], lo, hi)
    write_hist(os.path.join(a.outdir, f"{stem}.rescaled.hist"), k, lo, hi)
    write_hist(os.path.join(a.outdir, "reference.hist"), r, lo, hi)
    with open(os.path.join(a.outdir, f"{stem}.tradeoff.tsv"), "w") as f:
        f.write("M\tretained\tL1\n")
        for i in range(0, Ms.size, 20):
            f.write(f"{Ms[i]:.6g}\t{Rs[i]:.0f}\t{Es[i]:.6f}\n")
    kept = int(k.sum())
    p = k / max(kept, 1)
    L1 = float(np.abs(p - r).sum())
    cdf_gap = float(np.abs(np.cumsum(p) - np.cumsum(r)).max())
    miss = np.flatnonzero((k < np.floor(M * r)) & (r > 0))
    summary = {
        "query_records": N, "skipped_non_bed_lines": skipped,
        "query_records_outside_reference_range": int(q[LM]) + int(qL[r == 0].sum()),
        "rule": how, "scale_M": f"{M:.1f}", "retained": kept, "retained_fraction": f"{kept / max(N, 1):.4f}",
        "L1_rescaled_vs_reference": f"{L1:.4f}", "L1_query_vs_reference": f"{np.abs(qL / qL.sum() - r).sum():.4f}",
        "KS_max_cdf_gap": f"{cdf_gap:.4f}",
        "underfilled_lengths": f"{miss.size} ({miss.min()}-{miss.max()} bp)" if miss.size else "0",
        "seed": a.seed, "output": out_bed, "time_s": f"{t2 - t0:.2f} (pass1 {t1 - t0:.2f})",
    }
    with open(os.path.join(a.outdir, f"{stem}.summary.tsv"), "w") as f:
        for kk, v in summary.items():
            f.write(f"{kk}\t{v}\n")
    for kk, v in summary.items():
        print(f"{kk:40s} {v}")


if __name__ == "__main__":
    main()
