"""Conformal prediction wrappers for the vol-forecasting point models.

Modules:
    hqr: Heteroscedastic Quantile Regression + split-conformal calibration.
         Produces (PI_lo, PI_hi, spread) given a trained point model and a
         held-out calibration fold. Designed to address the quantile-LGBM
         failure from 2026-04-17 session where spread↔|err| Spearman
         was 0.06 << the 0.30 threshold needed for sizing.
"""
