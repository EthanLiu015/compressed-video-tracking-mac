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
is `Adaptive._signal`.
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

    def should_anchor(self, fmv: FrameMV) -> bool:
        intra_frac = 1.0 - float(fmv.occupancy.mean())
        self._since_anchor += 1

        fire = fmv.pict_type == "I" or self._since_anchor >= self.max_interval
        if not fire and self._warm and self._since_anchor >= self.min_interval:
            fire = intra_frac > self._ema * self.spike_factor and intra_frac > 0.05

        if not self._warm:
            self._ema = intra_frac
            self._warm = True
        else:
            self._ema = self.ema_alpha * intra_frac + (1 - self.ema_alpha) * self._ema

        if fire:
            self._since_anchor = 0
        return fire
