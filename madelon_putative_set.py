#!/usr/bin/env python3
"""Identify Madelon's putative relevant set (S1 Appendix): connected components of
the |Pearson r| > threshold graph over the 500 training columns."""
import sys
import numpy as np
import data_loader

X, y, _ = data_loader.load_dataset("madelon")
C = np.abs(np.corrcoef(X.to_numpy(float), rowvar=False)); np.fill_diagonal(C, 0)
cols = list(X.columns)


def components(th):
    adj = C > th; seen, out = set(), []
    for i in range(len(cols)):
        if i in seen or not adj[i].any():
            continue
        stack, comp = [i], set()
        while stack:
            j = stack.pop()
            if j not in comp:
                comp.add(j); stack += list(np.where(adj[j])[0])
        seen |= comp; out.append(sorted(cols[k] for k in comp))
    return out


for th in (0.15, 0.2, 0.3, 0.4, 0.5, 0.6):
    cs = components(th)
    print(th, sorted(len(c) for c in cs))
T = components(0.3)[0]
idx = [cols.index(c) for c in T]; rest = [i for i in range(len(cols)) if i not in idx]
print("set:", sorted(T, key=lambda s: int(s[1:])))
print("min within-set max |r|:", round(float(C[np.ix_(idx, idx)].max(1).min()), 3),
      " max |r| of any probe:", round(float(C[rest].max()), 3))
