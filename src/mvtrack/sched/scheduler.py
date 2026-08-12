"""Anchor schedulers: decide which frames get full decode + detector.

`FixedInterval` is the ablation baseline (every Nth frame, already used by
`eval.run.run_mv_fixed`). `Adaptive` fires anchors on signals that correlate
with real residual energy without decoding residual coefficients: a frame's
intra-coded block fraction (`1 - occupancy.mean()`) rises when the encoder
couldn't find a good motion match, which happens at scene cuts and around
complex/new motion — exactly where MV-propagated boxes drift most. True
residual-coefficient energy would be a stronger signal but PyAV/FFmpeg don't
expose decoded residuals through the Python side-data API; this proxy is
buildable now and directly testable, so it's the first cut. Swap-in point
is `Adaptive.urgency`.

`Adaptive.urgency()` exposes the same per-frame spike-ratio signal
`should_anchor()` decides on internally, as its own method -- added for the
global cross-stream scheduler (src/mvtrack/sched/global_budget.py), which
needs a per-stream "how urgent is this frame" score to rank concurrent
streams against each other, not just a local yes/no decision. Refactored
`should_anchor()` to call it rather than duplicate the EMA bookkeeping;
behavior is unchanged (verified via a before/after rerun of `eval/run.py
--pipeline mv-adaptive` on MOT17-02 -- identical anchor rate and HOTA).
"""

from dataclasses import dataclass, field

from mvtrack.extract import FrameMV


@dataclass
class FixedInterval:
    interval: int = 5

    def should_anchor(self, fmv: FrameMV) -> bool:
        return fmv.index % self.interval == 0


@dataclass
class Adaptive:
    """Anchor on scene cuts (I-frames), intra-fraction spikes vs. a rolling
    baseline, or a max-age safety net (so a static-but-driftless region of a
    still doesn't starve the tracker of any correction)."""

    min_interval: int = 2
    max_interval: int = 8
    spike_factor: float = 1.4
    ema_alpha: float = 0.2
    _ema: float = field(default=0.0, init=False)
    _since_anchor: int = field(default=0, init=False)
    _warm: bool = field(default=False, init=False)
    _last_intra_frac: float = field(default=0.0, init=False)

    _EPS = 1e-6

    def urgency(self, fmv: FrameMV) -> float:
        """Spike ratio: this frame's intra-coded fraction over the rolling
        EMA baseline as of BEFORE this frame (matches the comparison
        `should_anchor` has always made) -- >1 means "more intra-coded
        than this stream's own recent normal," a signal comparable across
        streams with different textures/baselines, unlike raw intra_frac.
        Mutates the same EMA/`_since_anchor` state `should_anchor` reads;
        call at most once per frame (`should_anchor` does this for you)."""
        intra_frac = 1.0 - float(fmv.occupancy.mean())
        self._last_intra_frac = intra_frac
        self._since_anchor += 1

        ratio = intra_frac / max(self._ema, self._EPS) if self._warm else 1.0

        if not self._warm:
            self._ema = intra_frac
            self._warm = True
        else:
            self._ema = self.ema_alpha * intra_frac + (1 - self.ema_alpha) * self._ema

        return ratio

    def should_anchor(self, fmv: FrameMV) -> bool:
        was_warm = self._warm
        ratio = self.urgency(fmv)
        intra_frac = self._last_intra_frac

        fire = fmv.pict_type == "I" or self._since_anchor >= self.max_interval
        if not fire and was_warm and self._since_anchor >= self.min_interval:
            fire = ratio > self.spike_factor and intra_frac > 0.05

        if fire:
            self._since_anchor = 0
        return fire
