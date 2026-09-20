#!/usr/bin/env python3
"""Recompute the mathematical coherence columns from the archived artifacts.

The coherence metric is a pure function of the CLIP embeddings and the narrative
paths, so it can be recomputed exactly, offline, with no API calls. This script
does that for the narrative set that was actually evaluated in the published
run, and reports what Table 1's "Min Coh." column and Table 2's correlations
should read.

Equation (1) of the paper:

    q(i, j) = sqrt( S(z_i, z_j) * T(p_i, p_j) )
    S(z_i, z_j) = 1 - arccos(cos_sim(z_i, z_j)) / pi

The experiment passes a single-column membership matrix
(`dummy_cluster_probs = np.ones((n, 1))`), so the cluster distributions are
uniform, T = 1, and q reduces to sqrt(S).

Three columns are produced:

  eq1_raw         The reported convention: S exactly as Equation (1) defines
                  it, unrescaled. This is the number to quote.
  eq1_normalised  The same, computed on the min-max rescaled table the linear
                  program uses for its edge weights. Strictly monotone with
                  respect to eq1_raw, so it preserves every ordering and rank
                  statistic, but its unit is pinned to the extreme pair of this
                  particular collection and so does not travel.
  published       What the published `min_coherence` column contains: the
                  rescaled similarity without the square root.

Usage:
    python reconstruct_coherence.py [--narratives PATH] [--out PATH]
"""

import argparse
import pickle
import statistics as st
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

SOURCES = ["human", "narrative_maps", "random"]


@dataclass
class ImageNarrative:
    """Mirrors the notebook dataclass so the cached narratives unpickle."""

    id: str
    source: str
    image_ids: List[str]
    image_paths: List[str]
    coherence_scores: Optional[Dict[str, float]] = None


def load_narratives(path):
    # The cache was pickled from the notebook's __main__, so the class has to be
    # reachable under that name.
    sys.modules["__main__"].ImageNarrative = ImageNarrative
    with open(path, "rb") as fh:
        cache = pickle.load(fh)
    return cache["narratives"], cache["all_image_ids"], np.asarray(cache["embeddings"])


def similarity_tables(embeddings):
    """Raw angular similarity, and its min-max rescaling over the off-diagonal."""
    units = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    raw = 1.0 - np.arccos(np.clip(units @ units.T, -1.0, 1.0)) / np.pi

    off_diagonal = ~np.eye(raw.shape[0], dtype=bool)
    lo, hi = raw[off_diagonal].min(), raw[off_diagonal].max()
    normalised = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
    return raw, normalised


def path_scores(path, table, apply_sqrt=True):
    """Minimum and average coherence along one path."""
    pairs = [table[a, b] for a, b in zip(path[:-1], path[1:])]
    values = np.sqrt(np.maximum(pairs, 0.0)) if apply_sqrt else np.asarray(pairs)
    return float(values.min()), float(values.mean())


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--narratives", default="published_run/vlm_narratives_cache.pkl")
    parser.add_argument("--judges", default="published_run/vlm_experiment_results.csv")
    parser.add_argument("--out", default="coherence_reconstructed.csv")
    args = parser.parse_args()

    narratives, image_ids, embeddings = load_narratives(args.narratives)
    position = {name: i for i, name in enumerate(image_ids)}
    raw, normalised = similarity_tables(embeddings)

    print(f"{len(narratives)} narratives | {len(image_ids)} images | embeddings {embeddings.shape}")

    variants = {
        "eq1_raw": (raw, True),                # sqrt(S), S as Equation (1) defines it
        "eq1_normalised": (normalised, True),  # sqrt(S) on the LP's rescaled table
        "published": (normalised, False),      # what the published column holds
    }

    rows = []
    for narrative in narratives:
        path = [position[i] for i in narrative.image_ids]
        row = {"narrative_id": narrative.id, "source": narrative.source, "length": len(path)}
        for name, (table, use_sqrt) in variants.items():
            row[f"min_{name}"], row[f"avg_{name}"] = path_scores(path, table, use_sqrt)
        rows.append(row)

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(",".join(rows[0]) + "\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row.values()) + "\n")
    print(f"per-narrative values written to {args.out}\n")

    # ---- Table 1, coherence columns -------------------------------------
    print("Table 1, minimum coherence (mean +/- sd over narratives)")
    print(f"{'source':16}{'N':>4}  " + "".join(f"{n:>22}" for n in variants))
    for source in SOURCES:
        group = [r for r in rows if r["source"] == source]
        cells = ""
        for name in variants:
            values = [r[f"min_{name}"] for r in group]
            cells += f"{st.mean(values):>15.3f} +/-{st.stdev(values):>5.3f}"
        print(f"{source:16}{len(group):>4}  {cells}")

    print("\nTable 1, average coherence (mean +/- sd over narratives)")
    print(f"{'source':16}{'N':>4}  " + "".join(f"{n:>22}" for n in variants))
    for source in SOURCES:
        group = [r for r in rows if r["source"] == source]
        cells = ""
        for name in variants:
            values = [r[f"avg_{name}"] for r in group]
            cells += f"{st.mean(values):>15.3f} +/-{st.stdev(values):>5.3f}"
        print(f"{source:16}{len(group):>4}  {cells}")

    # ---- Table 2, correlations with the judge scores --------------------
    try:
        import csv

        with open(args.judges, encoding="utf-8") as fh:
            judged = {r["narrative_id"]: r for r in csv.DictReader(fh)}
    except FileNotFoundError:
        print(f"\n(skipping Table 2: {args.judges} not found)")
        return

    paired = [(r, judged[r["narrative_id"]]) for r in rows if r["narrative_id"] in judged]
    print(f"\nTable 2, Pearson correlation with the judge scores (n = {len(paired)})")
    print(f"{'variant':18}{'caption/min':>14}{'caption/avg':>14}{'vlm/min':>14}{'vlm/avg':>14}")
    for name in variants:
        cells = ""
        for judge in ("caption_score_mean", "vlm_score_mean"):
            scores = [float(j[judge]) for _, j in paired]
            for stat in ("min", "avg"):
                cells += f"{pearson(scores, [r[f'{stat}_{name}'] for r, _ in paired]):>14.3f}"
        print(f"{name:18}{cells}")

    print(
        "\nNote: min-max rescaling is affine, so it leaves Pearson correlations\n"
        "unchanged; the square root does not, which is why the eq1_* rows differ\n"
        "from the published row."
    )

    # ---- does the corrected metric still rank the three sources? ---------
    from itertools import combinations

    from scipy import stats

    def values(source, column, dedupe):
        vals = [r[column] for r in rows if r["source"] == source]
        if dedupe and source == "narrative_maps":
            # The 10 replications per baseline were the same path, so they carry
            # identical coherence. Collapsing them gives the honest sample size.
            vals = sorted({round(v, 12) for v in vals})
        return vals

    for column in ("min_eq1_raw", "avg_eq1_raw"):
        print(f"\nDiscrimination on {column}")
        for dedupe in (False, True):
            label = "Narrative Maps collapsed to distinct paths" if dedupe else "as published"
            groups = {s: values(s, column, dedupe) for s in SOURCES}
            ranking = " > ".join(sorted(SOURCES, key=lambda s: -st.mean(groups[s])))
            sizes = ", ".join(f"{s}={len(groups[s])}" for s in SOURCES)
            print(f"  {label} ({sizes}): {ranking}")
            for (a, b) in combinations(SOURCES, 2):
                A, B = groups[a], groups[b]
                t, p = stats.ttest_ind(A, B)
                d = (st.mean(A) - st.mean(B)) / ((st.variance(A) + st.variance(B)) / 2) ** 0.5
                mark = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"      {a:15}vs {b:15} t={t:7.3f}  p={p:.2e}  d={d:6.2f}  {mark}")


if __name__ == "__main__":
    main()
