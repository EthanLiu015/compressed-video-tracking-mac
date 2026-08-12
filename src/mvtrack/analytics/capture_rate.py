"""Capture rate: of everyone whose path came near a zone, what fraction
actually stopped. The real retail metric -- raw dwell counts have no
denominator on their own (5 dwells out of 6 passersby is a completely
different result from 5 dwells out of 500), and this project's PULSE work so
far only ever reported the numerator.
"""

from .zones import Zone, point_near_zone


def classify_zone_traffic(tracks: dict, dwellers: dict, zone: Zone) -> dict:
    """Every track -> "stopped" (already in `dwellers`), "passed" (came
    within `zone.proximity_cm` of the zone without meeting the dwell
    criteria), or "unrelated" (trajectory never went near the zone at
    all -- excluded from the capture-rate denominator, since someone who
    never walked past a storefront was never a real capture opportunity)."""
    traffic = {}
    for tid, history in tracks.items():
        if tid in dwellers:
            traffic[tid] = "stopped"
            continue
        near = any(point_near_zone(p, zone) for _, p in history)
        traffic[tid] = "passed" if near else "unrelated"
    return traffic


def capture_rate(traffic: dict) -> float:
    """stopped / (stopped + passed). NaN if nobody ever passed the zone."""
    stopped = sum(1 for v in traffic.values() if v == "stopped")
    passersby = sum(1 for v in traffic.values() if v in ("stopped", "passed"))
    return stopped / passersby if passersby else float("nan")
