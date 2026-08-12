"""Synthetic correctness check for mvtrack.analytics.loitering -- CAVIAR's
real LeftBag scenario sits on a fisheye camera YOLOv8s can't detect people
on (confirmed directly, see loitering.py's module docstring), so this
module is validated against a synthetic fixture instead of real video, the
same kind of quick correctness check this project already uses for other
new numerical/logic components (e.g. MVTracker's re-association sanity
script)."""

import numpy as np

from mvtrack.analytics import detect_abandoned_objects, is_object_stationary

FRAME_DT = 1.0 / 25


def _static_history(pos, n_frames, start=0, jitter=0.0, rng=None):
    rng = rng or np.random.default_rng(0)
    return [
        (start + i, np.array(pos) + (rng.uniform(-jitter, jitter, size=2) if jitter else np.zeros(2)))
        for i in range(n_frames)
    ]


def _moving_history(start_pos, velocity, n_frames, start=0):
    return [(start + i, np.array(start_pos) + np.array(velocity) * i) for i in range(n_frames)]


def test_stationary_object_flagged_after_owner_leaves():
    # Object dropped at frame 0, stays exactly still for 300 frames (12s).
    obj = {0: _static_history((100.0, 100.0), 300)}
    # Owner stands next to it for the first 2s (50 frames), then walks away
    # and never returns.
    owner_near = _static_history((105.0, 100.0), 50, jitter=2.0)
    owner_leaving = _moving_history((110.0, 100.0), (20.0, 0.0), 250, start=50)
    persons = {0: owner_near + owner_leaving}

    events = detect_abandoned_objects(
        obj, persons, FRAME_DT, stationary_seconds=5.0, separation_cm=200.0, separation_seconds=3.0,
    )
    assert len(events) == 1, events
    assert events[0]["object_id"] == 0
    # Should flag once stationary_seconds have passed AND separation_seconds
    # have passed since the owner was last within separation_cm -- not
    # immediately at frame 0.
    assert events[0]["flagged_frame"] > 125, events  # 5s stationary alone is already frame 125


def test_object_not_flagged_while_owner_stays_nearby():
    obj = {0: _static_history((100.0, 100.0), 300)}
    owner = _static_history((105.0, 100.0), 300, jitter=2.0)  # always within separation_cm
    persons = {0: owner}

    events = detect_abandoned_objects(
        obj, persons, FRAME_DT, stationary_seconds=5.0, separation_cm=200.0, separation_seconds=3.0,
    )
    assert events == [], events


def test_moving_object_never_flagged():
    # A bag being carried, not a bag left behind -- real motion the whole time.
    obj = {0: _moving_history((0.0, 0.0), (15.0, 0.0), 300)}
    persons = {0: _moving_history((5.0, 0.0), (15.0, 0.0), 300)}

    events = detect_abandoned_objects(
        obj, persons, FRAME_DT, stationary_seconds=5.0, separation_cm=200.0, separation_seconds=3.0,
    )
    assert events == [], events
    assert not is_object_stationary(obj[0])


def test_jitter_floor_rejects_exact_static_object_only_when_appropriate():
    # A perfectly static object (0 jitter) IS stationary -- that's the real
    # signature this module is built to catch (unlike the dwell-radius
    # jitter floor elsewhere in this project, which exists to REJECT
    # exact-static false positives among PERSON tracks -- opposite
    # direction, different track type, intentionally not reused here).
    obj = {0: _static_history((50.0, 50.0), 200)}
    assert is_object_stationary(obj[0])


if __name__ == "__main__":
    test_stationary_object_flagged_after_owner_leaves()
    test_object_not_flagged_while_owner_stays_nearby()
    test_moving_object_never_flagged()
    test_jitter_floor_rejects_exact_static_object_only_when_appropriate()
    print("all loitering synthetic checks passed")
