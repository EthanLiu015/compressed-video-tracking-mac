"""Named world-space regions ("exhibit zones", a storefront, a rug) and
membership/proximity tests against them -- shared by every PULSE fusion
script instead of each one hand-rolling its own polygon check (as
scripts/epfl_lab_fusion.py originally did, the only place `point_in_polygon`
existed before this move)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Zone:
    name: str
    polygon: np.ndarray  # Nx2 world-space (cm) polygon vertices
    proximity_cm: float = 0.0  # extra buffer beyond the polygon edge still counted "near" this zone


def point_in_polygon(pt, poly) -> bool:
    """Standard ray-casting point-in-polygon test."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def _point_segment_dist(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = np.clip(np.dot(pt - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(pt - proj))


def distance_to_polygon(pt, poly) -> float:
    """0.0 if `pt` is inside `poly`; otherwise the distance to the nearest edge."""
    if point_in_polygon(pt, poly):
        return 0.0
    poly = np.asarray(poly, dtype=float)
    pt = np.asarray(pt, dtype=float)
    n = len(poly)
    return min(_point_segment_dist(pt, poly[i], poly[(i + 1) % n]) for i in range(n))


def point_near_zone(pt, zone: Zone) -> bool:
    """Inside the zone, or within its `proximity_cm` buffer of the edge."""
    return distance_to_polygon(pt, zone.polygon) <= zone.proximity_cm


def track_zone(track_history, zone: Zone) -> str:
    """'inside' if any point of this track's trajectory ever fell inside
    `zone.polygon`, 'near' if it came within `zone.proximity_cm` without
    ever entering, else 'outside'."""
    xy = np.array([p for _, p in track_history])
    dists = np.array([distance_to_polygon(p, zone.polygon) for p in xy])
    if (dists <= 0.0).any():
        return "inside"
    if (dists <= zone.proximity_cm).any():
        return "near"
    return "outside"
