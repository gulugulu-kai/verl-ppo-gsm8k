#!/usr/bin/env python3
"""Compare several GSM8K eval jsonl files produced by eval_gsm8k.py.

Reports, per run:
  - strict accuracy  (the PPO reward metric: requires `#### <answer>` AND correct value)
  - lenient accuracy (heuristic: last number in the text vs ground truth)
  - format rate      (share of outputs containing `####`)
  - Wilson 95% CI for both accuracies

And, for every run after the first, a PAIRED comparison against the first run
(same 1319 questions, so McNemar's exact test applies) on both metrics.

Usage:
  python3 compare_evals.py --run base=/root/eval_out/base.jsonl --run ep1=/root/eval_out/trained.jsonl
"""
import argparse
import json
import math
import re

FMT_RE = re.compile(r'####\s*([^\n]+)')
NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')


def last_number(s):
    nums = NUM_RE.findall(s.replace(',', ''))
    return nums[-1] if nums else None


def score_row(r):
    gt = float(r['ground_truth'])
    out = r['output']
    m = FMT_RE.search(out)
    strict = float(r.get('score', 0.0)) > 0
    fmt = m is not None
    cand = last_number(m.group(1)) if fmt else None
    if cand is None:
        cand = last_number(out)
    lenient = False
    try:
        lenient = cand is not None and abs(float(cand) - gt) < 1e-6
    except Exception:
        lenient = False
    return strict, lenient, fmt


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar (binomial) test. b/c = discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_run(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    strict, lenient, fmt = [], [], []
    for r in rows:
        s, l, f = score_row(r)
        strict.append(s)
        lenient.append(l)
        fmt.append(f)
    return {'path': path, 'n': len(rows), 'strict': strict, 'lenient': lenient, 'fmt': fmt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True, help='tag=path (repeatable)')
    ap.add_argument('--out-md', default=None)
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        tag, _, path = spec.partition('=')
        runs.append((tag, load_run(path)))

    lines = []
    lines.append('| run | n | strict acc | strict 95% CI | lenient acc | lenient 95% CI | format rate |')
    lines.append('|---|---|---|---|---|---|---|')
    for tag, r in runs:
        n = r['n']
        s = sum(r['strict'])
        l = sum(r['lenient'])
        f = sum(r['fmt'])
        sl, sh = wilson(s, n)
        ll, lh = wilson(l, n)
        lines.append('| %s | %d | **%.2f%%** | %.2f–%.2f%% | **%.2f%%** | %.2f–%.2f%% | %.2f%% |' % (
            tag, n, 100 * s / n, 100 * sl, 100 * sh, 100 * l / n, 100 * ll, 100 * lh, 100 * f / n))

    base_tag, base = runs[0]
    lines.append('')
    lines.append('Paired comparisons vs `%s` (exact McNemar on the same questions):' % base_tag)
    lines.append('')
    lines.append('| pair | metric | base ok | other ok | only base | only other | delta | p (exact) |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for tag, r in runs[1:]:
        assert r['n'] == base['n'], 'runs must share the same question set'
        for metric in ('strict', 'lenient'):
            b = sum(1 for x, y in zip(base[metric], r[metric]) if (not x) and y)   # only other ok
            c = sum(1 for x, y in zip(base[metric], r[metric]) if x and (not y))   # only base ok
            p = mcnemar_exact(b, c)
            bo = sum(base[metric])
            oo = sum(r[metric])
            n = base['n']
            lines.append('| %s vs %s | %s | %d (%.2f%%) | %d (%.2f%%) | %d | %d | %+.2f pp | %.2g |' % (
                base_tag, tag, metric, bo, 100 * bo / n, oo, 100 * oo / n, c, b,
                100 * (oo - bo) / n, p))

    out = '\n'.join(lines)
    print(out)
    if args.out_md:
        with open(args.out_md, 'w', encoding='utf-8') as f:
            f.write(out + '\n')
        print('\n[compare] written to %s' % args.out_md, flush=True)


if __name__ == '__main__':
    main()
