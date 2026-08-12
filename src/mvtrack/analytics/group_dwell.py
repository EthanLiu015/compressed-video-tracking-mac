"""Group/social behavior: are two (or more) people moving/stopping together,
not just independently near the same zone. Real value: distinguishes an
engaged pair/group from a solo quick glance -- directly connects back to the
Whyte plaza-quality research PULSE's own scripts already cite (social
clustering as a core signal of a space's quality, not just raw occupancy).
"""

import numpy as np


def find_concurrent_pairs(tracks: dict) -> list:
    """All (tid_a, tid_b, (overlap_start_frame, overlap_end_frame)) for
    tracks whose frame ranges overlap at all."""
    items = list(tracks.items())
    pairs = []
    for i in range(len(items)):
        tid_a, hist_a = items[i]
        frames_a = [f for f, _ in hist_a]
        lo_a, hi_a = min(frames_a), max(frames_a)
        for j in range(i + 1, len(items)):
            tid_b, hist_b = items[j]
            frames_b = [f for f, _ in hist_b]
            lo_b, hi_b = min(frames_b), max(frames_b)
            overlap_lo, overlap_hi = max(lo_a, lo_b), min(hi_a, hi_b)
            if overlap_lo <= overlap_hi:
                pairs.append((tid_a, tid_b, (overlap_lo, overlap_hi)))
    return pairs


def pair_proximity_fraction(hist_a, hist_b, overlap: tuple, proximity_cm: float) -> float:
    """Of the frames both tracks were alive during `overlap`, what fraction
    were the two positions within `proximity_cm` of each other."""
    lo, hi = overlap
    pos_a = {f: p for f, p in hist_a}
    pos_b = {f: p for f, p in hist_b}
    common = [f for f in range(lo, hi + 1) if f in pos_a and f in pos_b]
    if not common:
        return 0.0
    dists = [np.linalg.norm(np.asarray(pos_a[f]) - np.asarray(pos_b[f])) for f in common]
    return sum(1 for d in dists if d <= proximity_cm) / len(dists)


def find_companion_pairs(tracks: dict, proximity_cm: float = 150.0, min_fraction_together: float = 0.6) -> list:
    """Every concurrent pair that stayed within `proximity_cm` for at least
    `min_fraction_together` of their shared time on screen -- "moving/
    standing together," independent of whether either one ever dwells.
    This is the core companionship signal (used directly by Meet_WalkTogether-
    style validation, where two people never necessarily stop at all)."""
    companions = []
    for tid_a, tid_b, overlap in find_concurrent_pairs(tracks):
        frac = pair_proximity_fraction(tracks[tid_a], tracks[tid_b], overlap, proximity_cm)
        if frac >= min_fraction_together:
            companions.append((tid_a, tid_b, frac))
    return companions


def classify_group_dwell(tracks: dict, dwellers: dict, proximity_cm: float = 150.0,
                          min_fraction_together: float = 0.6) -> dict:
    """dweller tid -> True if it was part of a companion pair (see
    `find_companion_pairs`) during its dwell window -- i.e. it stopped
    together with someone else, not alone."""
    grouped = {tid: False for tid in dwellers}
    for tid_a, tid_b, _frac in find_companion_pairs(tracks, proximity_cm, min_fraction_together):
        if tid_a in dwellers:
            grouped[tid_a] = True
        if tid_b in dwellers:
            grouped[tid_b] = True
    return grouped
