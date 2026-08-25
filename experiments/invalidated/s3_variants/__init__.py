"""Invalidated S3 variant experiments preserved as forensic source.

The D family estimates thresholds from full-sample terminal outcomes and
rewrites already-realized trade PnL. The H family composes those transforms
and applies additional terminal-PnL caps and boosts without a causal event
path. These operations are non-causal. They are not strategy implementations,
executable trading rules, or valid performance evidence.

The remaining A/B/C/E/F/G modules are retained only to reconstruct the study
that consumed those invalid transforms. Nothing in this package is production
strategy code.
"""
