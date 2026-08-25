"""Historical milestone-6 evaluation report.

UNTRUSTED LEGACY EXPERIMENT: this program hard-codes early experiment outputs
and a contemporaneous deployment recommendation. It is preserved as research
provenance only; its claims are not current and are superseded by
``docs/VALIDATION.md``.

Compiles all experiment results into a single comparison table.
Records what the project considered deploying at that time.

Compares against the plan's success/kill criteria:
  - Success: net-of-cost Sharpe > 0.5, RankIC > 0.03, DSR > 0.95
  - Kill: no model beats SARIMA after costs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def compile_results(results_dir: str | Path = "results") -> dict:
    """
    Compile all experimental findings into a structured report.
    """
    results_dir = Path(results_dir)

    report = {
        "generated": datetime.now().isoformat(),
        "findings": {},
    }

    # Track A: Direction Prediction
    report["findings"]["direction_prediction"] = {
        "verdict": "FAILS",
        "evidence": {
            "BTC_15m": {"accuracy": 0.476, "p_value": 1.0, "n": 401},
            "ETH_15m": {"accuracy": 0.475, "p_value": 1.0, "n": 1363},
            "SOL_15m": {"accuracy": 0.495, "p_value": 1.0, "n": 1942},
            "BTC_1H": {"accuracy": 0.493, "p_value": 1.0, "n": 471},
            "BTC_4H": {"accuracy": 0.500, "p_value": 1.0, "n": 142},
        },
        "conclusion": (
            "Direction prediction fails at ALL timeframes (15m/1H/4H), ALL assets "
            "(BTC/ETH/SOL), ALL feature sets (13 and 23 features). Consistent with "
            "literature (Kolm 2023: alpha decays within ~2 price changes)."
        ),
    }

    # Track B: Meta-Labeling
    report["findings"]["meta_labeling"] = {
        "verdict": "FAILS (with weak primary signal)",
        "evidence": {
            "BTC_15m_crash": {
                "meta_f1": 0.457,
                "meta_accuracy": 0.474,
                "primary_sharpe": -33.4,
                "filtered_sharpe": -29.1,
                "sized_sharpe": -22.7,
            },
        },
        "conclusion": (
            "Meta-labeling cannot rescue a fundamentally bad primary signal. "
            "Crash at -0.5% on 15m is too frequent and noisy. Need event-driven "
            "primary with >52% baseline accuracy."
        ),
    }

    # Track C: Volatility Prediction
    report["findings"]["volatility_prediction"] = {
        "verdict": "SUCCEEDS",
        "evidence": {
            "BTC_15m": {"rank_ic": 0.605, "improvement": 0.162, "n": 2003},
            "ETH_15m": {"rank_ic": 0.574, "improvement": 0.112, "n": 6816},
            "SOL_15m": {"rank_ic": 0.550, "improvement": 0.121, "n": 9710},
            "BTC_1H": {"rank_ic": 0.382, "improvement": 0.244, "n": 2353},
            "BTC_4H": {"rank_ic": 0.049, "improvement": 0.082, "n": 706},
            "vol_sizer_production": {"rank_ic": 0.679, "improvement": 0.305},
        },
        "conclusion": (
            "Volatility prediction succeeds across ALL assets at 15m (RankIC 0.55-0.68). "
            "6 consensus features identified (MDI∩MDA∩SFI): bar_range, parkinson_vol, "
            "daily_vol, vol_ratio, vol_zscore, hour_sin. Finer timeframes give stronger signal. "
            "Production vol sizer achieves +30.5% improvement with confidence thresholding."
        ),
    }

    # Kronos Zero-Shot
    report["findings"]["kronos_zero_shot"] = {
        "verdict": "FAILS (expected — not trained on crypto)",
        "evidence": {
            "mini_BTC_15m_n200": {"accuracy": 0.440},
            "base_BTC_15m_n200": {"accuracy": 0.550, "p_value": 0.089},
        },
        "conclusion": "Pre-trained on equities. Fine-tuning running. Base at 55% is borderline.",
    }

    # Feature Importance
    report["findings"]["feature_importance"] = {
        "method": "MDI ∩ MDA ∩ SFI consensus",
        "consensus_features": [
            "bar_range", "parkinson_vol", "daily_vol",
            "vol_ratio", "vol_zscore", "hour_sin",
        ],
        "in_2_of_3": ["cs_spread", "dow_sin", "ret_1"],
    }

    # Data Quality
    report["findings"]["data_quality"] = {
        "verdict": "EXCELLENT",
        "timestamp_gaps": 0,
        "nan_pct": 0.0,
        "ohlc_violations": 0,
        "kurtosis": 108.23,
        "fix_applied": "Winsorization at 1st/99th percentile",
    }

    # Success Criteria Assessment
    report["success_criteria"] = {
        "direction_beats_random": {"required": ">52% with p<0.05", "achieved": "NO"},
        "vol_rankic_gt_003": {"required": ">0.03", "achieved": "YES (0.55-0.68)"},
        "net_sharpe_gt_05": {"required": ">0.5", "achieved": "TBD (vol sizer not yet backtested as strategy)"},
        "dsr_gt_095": {"required": ">0.95", "achieved": "TBD"},
    }

    # Recommendation
    report["recommendation"] = {
        "deploy": "Vol-based position sizer for existing event-driven strategies",
        "architecture": (
            "Event signal (unlock/crash) → direction. "
            "Vol model (XGBoost, 6 features) → position size. "
            "Confidence threshold → only adjust when model is confident."
        ),
        "kill": [
            "Track A (direction prediction)",
            "Track B (meta-labeling with weak primary)",
        ],
        "continue": [
            "Kronos fine-tuning (check after completion)",
            "5m/1m vol testing (finer scales)",
            "Funding rate features (Ackerer insight)",
        ],
    }

    return report


def print_report(report: dict):
    """Print a human-readable summary."""
    print("=" * 70)
    print("ML TRADING SYSTEM — FINAL EVALUATION REPORT")
    print(f"Generated: {report['generated']}")
    print("=" * 70)

    for name, finding in report["findings"].items():
        verdict = finding.get("verdict", "N/A")
        print(f"\n--- {name.upper()} ---")
        print(f"  Verdict: {verdict}")
        if "conclusion" in finding:
            print(f"  {finding['conclusion']}")

    print("\n--- SUCCESS CRITERIA ---")
    for crit, status in report["success_criteria"].items():
        print(f"  {crit}: {status['required']} → {status['achieved']}")

    print("\n--- RECOMMENDATION ---")
    print(f"  Deploy: {report['recommendation']['deploy']}")
    print(f"  Kill: {', '.join(report['recommendation']['kill'])}")
    print(f"  Continue: {', '.join(report['recommendation']['continue'])}")


if __name__ == "__main__":
    report = compile_results()
    print_report(report)
    # Save
    with open("results/final_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
