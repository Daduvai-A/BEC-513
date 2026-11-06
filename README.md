# BEC-513 project-3

 V-plot matrix 

Reads the site-fragment pairs from STANDARD INPUT (or from files given as arguments) and writes
the count matrix for Gnuplot:

  wide : <prefix>.wide.tsv   gnuplot "nonuniform matrix" layout
           row 0      : <number of offsets>  -500 -499 ... 500
           rows 1..   : <fragment length>    count@-500 count@-499 ... count@500
         (i.e. 1st column = fragment length, following columns = counts per offset)
  long : <prefix>.long.tsv   "offset length count" (x y z), one line per cell

Input columns (1-based, defaults match the provided mapped.bed.gz):
  site window start/end = 3/4, site strand = 7, fragment start/end = 9/10.
offset = midpoint(fragment) - centre(site window), sign flipped for '-' strand sites so the
plot is oriented 5'->3' along the motif.  length = fragment end - start.

Usage:
  zcat mapped.bed.gz | python vplot.py -o vplot            # standard input (as required)
  python vplot.py -o vplot mapped.bed.gz                   # same, with parallel gzip decoding
