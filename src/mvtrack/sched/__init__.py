"""Adaptive anchor scheduler (weeks 11-12): decides when to fire full
decode + inference based on residual energy, scene cuts, and track drift.
Fixed-interval anchoring lives here too as the ablation baseline."""
