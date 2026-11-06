# Project 2: Rescale to Reference

This pipeline subsamples a query BED file (cfDNA fragments) so that the fragment-length distribution of the kept lines matches a reference distribution. It writes the subsampled BED file, the histograms and summary statistics, and the comparison plot (gnuplot, plus a matplotlib copy).

Result on the provided data (`query.bed.gz` with 13,835,737 fragments, against `reference.hist`):

| | value |
|---|---|
| fragments kept | **4,279,385** (30.9%) |
| L1 distance to the reference, before → after | 0.960 → **0.067** |
| maximum CDF gap (KS statistic) after | 0.032 |
| lengths where the query has too few fragments ("points that miss out") | 172–567 bp, mainly visible at 300–400 bp |
| run time (24 threads, WSL2) | 2.6 s (rescaling) + about 1 s (plots) |

![rescaled](example_output/query.rescaled.gnuplot.png)

## Run
```bash
./setup.sh                                                     # Python venv (numpy, numba, rapidgzip, isal, matplotlib)
./rescale_to_reference.sh -q query.bed.gz -r reference.hist -o rescaled_output
bash verify.sh query.bed.gz rescaled_output/query.rescaled.bed.gz rescaled_output/query.rescaled.hist reference.hist
```
Options:

| flag | meaning |
|---|---|
| `-t TOL` | keep as many fragments as possible with L1 ≤ TOL |
| `-n N` | keep about N fragments |
| `-s SEED` | random seed (default 0) |
| `-j THREADS` | number of threads |
| `-m GB` | memory budget (default: 50% of available RAM) |

Outputs in `OUTDIR` (`<stem>` is the query file name without `.bed.gz`, e.g. `query`):

| file | content |
|---|---|
| `<stem>.rescaled.bed.gz` | the subsampled BED. The lines are unchanged and stay in input order. |
| `<stem>.query.hist`, `<stem>.rescaled.hist`, `reference.hist` | `length  normalized_frequency  [count]` |
| `<stem>.summary.tsv` | counts, the selection rule, L1, KS, and the lengths that fall short |
| `<stem>.tradeoff.tsv` / `.png` | retained fragments against L1 error, for every scale M |
| `<stem>.rescaled.gnuplot.png`, `<stem>.rescaled.png` | the figure (gnuplot / matplotlib) |

## Method
Notation:
- q(ℓ) is the number of query fragments of length ℓ = end − start.
- r(ℓ) is the reference frequency, normalized to sum to 1.

Keeping k(ℓ) ≤ q(ℓ) fragments of length ℓ gives the distribution k(ℓ)/Σk.

1. **Exact match is too costly.**
   - Matching exactly needs k(ℓ) = M·r(ℓ) for every ℓ, so M ≤ M₀ = minₗ q(ℓ)/r(ℓ).
   - Here M₀ is limited by ℓ = 356 bp, where the query is sparse. The resulting sample has only 1.22 M fragments (8.8%).
2. **Relaxed target.**
   - Use k_M(ℓ) = min(q(ℓ), M·r(ℓ)). Every length matches the reference exactly, except those where the query has too few fragments; those are kept in full. These are the "points that miss out" mentioned in the assignment.
   - Both the number retained, R(M), and the error, L1(M), increase with M. `<stem>.tradeoff.png` shows the curve.
3. **Choosing M (default).**
   - Take the **knee** of the retained-vs-error curve (Kneedle method): the M that maximizes the normalized gain in retained fragments minus the normalized growth in error.
   - On this data it lands at 4.28 M fragments with L1 = 0.067. That agrees with the example figure in the assignment (n = 4,295,020).
   - `-t` and `-n` replace the knee with an explicit tolerance or target count.
4. **Exact-count sampling.**
   - For each length ℓ, exactly k(ℓ) of its q(ℓ) fragments are drawn uniformly at random without replacement (seeded).
   - Each fragment is identified by its ordinal among the query fragments of the same length, so the result does not depend on block size or thread count.
   - The output histogram therefore equals k(ℓ) exactly, with no extra sampling noise from Bernoulli thinning.
5. **Speed and memory.**
   - Two streaming passes over the input, one to build the histogram and one to select lines, both in O(N).
   - Parsing and selection are parallel Numba kernels, and gzip is decoded in parallel by rapidgzip.
   - Memory is bounded by the block size plus one bit per fragment: 1.5 GB by default, or 0.37 GB with `-m 0.25`, which still gives identical output.

## Verification (`verify.sh`, independent awk/sort implementation)
On the provided data, `verify.sh` confirmed that:
- the histogram of the output BED equals `rescaled.hist` exactly;
- every output line occurs in the query (a multiset subset check with `sort` and `comm`);
- the L1 distance between the output and the reference is 0.0671, the same as the reported value.

Separately, I checked by hand that:
- the same seed gives identical output with 8 and with 24 threads, and with a 0.25 GB budget;
- the alternative rules behave as expected: `-t 0.02` keeps 2.14 M fragments at L1 = 0.020, and `-n 1220959` keeps 1.22 M at L1 = 0.0004.
