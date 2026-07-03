"""Throughput / energy / multi-stream benchmark harness (weeks 11-12).
Aggregate fps across N parallel streams; energy via `powermetrics`."""

from .harness import run_multistream

__all__ = ["run_multistream"]
