"""HR-1 regression locks for the cross-horizon roll-up ranking.

The pre-HR-1 rollup ranked horizons by ABSOLUTE QLIKE / OOS-R². That is
wrong: target variance spans ~1.9× across horizons, so "lowest QLIKE" just
flags the lowest-variance target, not the best forecast. HR-1 re-ranks on
downstream NC-Sharpe (PRIMARY) + RankIC (scale-invariant) and SELECTS on
worst-year NC-Sharpe, reading the per-year arrays
the shards already emit.

These tests construct synthetic shards engineered so the QLIKE winner and
the NC-Sharpe winner are DIFFERENT horizons, then assert the rollup output
selects on NC-Sharpe / worst-year NC-Sharpe — never on QLIKE.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.aws.rollup_horizons as rollup


def _shard(h, *, qlike, nc_sharpe, per_year_nc, rank_ic, per_year_ic, oos_r2=0.5):
    """Build a minimal self-describing shard dict like run_horizon_job emits."""
    return {
        "horizon_minutes": h,
        "forecast": {
            "qlike_oos_mean": qlike,
            "oos_r2_pooled": oos_r2,
            "rank_ic_mean": rank_ic,
            "per_year": [
                {"year": y, "rank_ic": ic} for y, ic in per_year_ic
            ],
        },
        "downstream": {
            "status": "ok",
            "nc_sharpe": nc_sharpe,
            "nc_pnl_usd": nc_sharpe * 1e5,
            "n_trades": 1000,
            "per_year_nc_sharpe": [
                {"year": y, "nc_sharpe": s, "nc_pnl_usd": s * 1e4, "n_trades": 100}
                for y, s in per_year_nc
            ],
        },
    }


def _write_shards(root: Path, shards):
    for s in shards:
        d = root / f"results/h{s['horizon_minutes']}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "shard.json").write_text(json.dumps(s))


def _build_divergent_shards():
    # h=30 has the LOWEST QLIKE (would win the OLD ranking) but a POOR
    # worst-year NC-Sharpe (collapses in 2024).
    # h=60 has a WORSE (higher) QLIKE but the BEST + most-stable NC-Sharpe.
    return [
        _shard(
            30, qlike=0.50, nc_sharpe=3.0,
            per_year_nc=[(2022, 4.0), (2023, 3.5), (2024, -1.0)],  # worst = -1.0
            rank_ic=0.10,
            per_year_ic=[(2022, 0.12), (2023, 0.11), (2024, -0.05)],
        ),
        _shard(
            60, qlike=0.85, nc_sharpe=5.0,
            per_year_nc=[(2022, 4.5), (2023, 5.5), (2024, 4.0)],   # worst = 4.0
            rank_ic=0.18,
            per_year_ic=[(2022, 0.17), (2023, 0.19), (2024, 0.16)],
        ),
        _shard(
            240, qlike=0.95, nc_sharpe=2.0,
            per_year_nc=[(2022, 2.5), (2023, 2.0), (2024, 1.5)],
            rank_ic=0.14,
            per_year_ic=[(2022, 0.15), (2023, 0.13), (2024, 0.14)],
        ),
    ]


def test_rollup_selects_on_nc_sharpe_not_qlike(tmp_path, capsys):
    """The SELECT line must point at the worst-year NC-Sharpe winner
    (h=60), NOT the lowest-QLIKE horizon (h=30)."""
    _write_shards(tmp_path, _build_divergent_shards())
    # Drive main() via argv.
    import sys
    argv = sys.argv
    sys.argv = ["rollup_horizons.py", "--dir", str(tmp_path)]
    try:
        ret = rollup.main()
    finally:
        sys.argv = argv
    assert ret == 0
    out = capsys.readouterr().out

    # The PRIMARY selection is worst-year NC-Sharpe → h=60 (worst-yr 4.0).
    assert "SELECT" in out
    sel_line = next(l for l in out.splitlines() if "SELECT" in l and "WORST-YEAR" in l)
    assert "h=60m" in sel_line, (
        f"SELECT must pick the best worst-year NC-Sharpe horizon (h=60), "
        f"got line: {sel_line!r}"
    )
    # h=30 is the lowest-QLIKE horizon — it must NOT be the selection.
    assert "h=30m" not in sel_line


def test_rollup_does_not_emit_a_qlike_ranking_winner(tmp_path, capsys):
    """QLIKE / OOS-R² are diagnostics ONLY — the output must NOT present a
    'lowest QLIKE' or 'highest OOS R^2' as a selection/ranking pointer."""
    _write_shards(tmp_path, _build_divergent_shards())
    import sys
    argv = sys.argv
    sys.argv = ["rollup_horizons.py", "--dir", str(tmp_path)]
    try:
        rollup.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    # The old ranking pointers are gone.
    assert "lowest QLIKE" not in out
    assert "highest OOS R^2" not in out
    # And the diagnostic disclaimer is present.
    assert "NON-RANKING diagnostic" in out


def test_rollup_pooled_nc_sharpe_secondary(tmp_path, capsys):
    """Pooled NC-Sharpe is reported as a secondary rank-on metric and
    points at the pooled max (h=60, nc_sharpe 5.0)."""
    _write_shards(tmp_path, _build_divergent_shards())
    import sys
    argv = sys.argv
    sys.argv = ["rollup_horizons.py", "--dir", str(tmp_path)]
    try:
        rollup.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    pooled = next(l for l in out.splitlines() if "pooled NC-Sharpe" in l)
    assert "h=60m" in pooled


def test_rollup_handles_missing_per_year_gracefully(tmp_path, capsys):
    """A shard with no per_year arrays (e.g. a failed/empty downstream)
    must not crash the rollup — worst-year columns degrade to n/a."""
    shards = [
        {
            "horizon_minutes": 15,
            "forecast": {"qlike_oos_mean": 0.8, "oos_r2_pooled": 0.5,
                         "rank_ic_mean": 0.15},
            "downstream": {"status": "empty", "nc_sharpe": float("nan"),
                           "n_trades": 0},
        },
    ]
    _write_shards(tmp_path, shards)
    import sys
    argv = sys.argv
    sys.argv = ["rollup_horizons.py", "--dir", str(tmp_path)]
    try:
        ret = rollup.main()
    finally:
        sys.argv = argv
    assert ret == 0
    out = capsys.readouterr().out
    assert "15m" in out
