"""
alerts/alert_generator.py
--------------------------
Generates structured, human-readable risk alerts for compliance teams.

Each alert contains:
  - user_id, timestamp, risk_level (LOW/MEDIUM/HIGH/CRITICAL)
  - final_score, component scores
  - summary: one-line human readable description
  - findings: list of specific detected patterns
  - recommended_action: what compliance should do
  - top_features: raw feature names driving the score
  - raw_score_breakdown: dict of all component scores
"""

from datetime import datetime
from typing import Optional
import pandas as pd


# ── Risk levels ───────────────────────────────────────────────────────────────

def _risk_level(score: float) -> str:
    if score >= 0.85: return "CRITICAL"
    if score >= 0.70: return "HIGH"
    if score >= 0.55: return "MEDIUM"
    return "LOW"


# ── Behavior pattern detectors ────────────────────────────────────────────────

def _detect_portal_patterns(portal_events: list) -> list[str]:
    """Detect specific suspicious patterns from raw portal events."""
    findings = []
    if not portal_events:
        return findings

    df = pd.DataFrame(portal_events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    logins      = df[df["event_type"] == "login"]
    deposits    = df[df["event_type"] == "deposit"]
    withdrawals = df[df["event_type"] == "withdrawal"]
    kyc         = df[df["event_type"] == "kyc_change"]

    # Multi-IP burst
    if len(logins) >= 3:
        for _, window_start in logins["timestamp"].items():
            window_end = window_start + pd.Timedelta(minutes=60)
            in_window  = logins[(logins["timestamp"] >= window_start) &
                                (logins["timestamp"] <= window_end)]
            n_ips = in_window["ip_address"].nunique()
            if n_ips >= 4:
                findings.append(
                    f"Multi-IP login burst: {n_ips} distinct IPs within 60 minutes"
                )
                break

    # Brute force
    failed = logins[logins["status"] == "failed"]
    if len(failed) >= 5:
        findings.append(
            f"Brute force pattern: {len(failed)} failed login attempts detected"
        )

    # 3 AM logins
    if len(logins):
        night_logins = logins[logins["timestamp"].dt.hour.between(1, 4)]
        if len(night_logins) >= 3:
            findings.append(
                f"Unusual login hours: {len(night_logins)} logins between 1–4 AM"
            )

    # Structuring
    if len(deposits):
        dep_amts     = deposits["amount"].dropna()
        small_deps   = dep_amts[dep_amts < 500]
        if len(small_deps) >= 10:
            findings.append(
                f"Structuring pattern: {len(small_deps)} deposits under $500 "
                f"(total ${small_deps.sum():,.0f})"
            )

    # Deposit → withdraw abuse
    if len(deposits) and len(withdrawals):
        total_dep = deposits["amount"].dropna().sum()
        total_wd  = withdrawals["amount"].dropna().sum()
        ratio     = total_wd / max(total_dep, 1)
        if ratio > 0.75 and total_dep > 1000:
            findings.append(
                f"Deposit-withdrawal cycle: withdrew {ratio*100:.0f}% of deposits "
                f"(${total_wd:,.0f} of ${total_dep:,.0f})"
            )

    # KYC before withdrawal
    if len(kyc) and len(withdrawals):
        for wts in withdrawals["timestamp"]:
            pre_kyc = kyc[
                (kyc["timestamp"] >= wts - pd.Timedelta(hours=4)) &
                (kyc["timestamp"] <= wts)
            ]
            if len(pre_kyc) >= 3:
                wd_amt = withdrawals[withdrawals["timestamp"] == wts]["amount"].values
                amt_str = f"${wd_amt[0]:,.0f}" if len(wd_amt) else "unknown amount"
                findings.append(
                    f"KYC manipulation: {len(pre_kyc)} profile changes in 4h "
                    f"before {amt_str} withdrawal"
                )
                break

    return findings


def _detect_trading_patterns(trading_events: list) -> list[str]:
    """Detect suspicious trading patterns from raw trade events."""
    findings = []
    if not trading_events:
        return findings

    df = pd.DataFrame(trading_events)
    df["open_ts"] = pd.to_datetime(df["open_ts"])

    n        = len(df)
    lots     = df["lot_size"]
    pnl      = df["pnl"]
    dur      = df["duration_min"]
    instrs   = df["instrument"]

    # Volume spike
    if n > 5:
        daily_counts = df.groupby(df["open_ts"].dt.date).size()
        mean_daily   = daily_counts.mean()
        max_daily    = daily_counts.max()
        if max_daily > mean_daily * 5 and max_daily > 20:
            findings.append(
                f"Volume spike: {int(max_daily)} trades in one day "
                f"(avg {mean_daily:.1f}/day) — {max_daily/max(mean_daily,1):.1f}x baseline"
            )

    # Lot size spike
    if n > 5:
        mu, sigma = lots.mean(), lots.std()
        if sigma > 0:
            max_z = ((lots - mu) / sigma).max()
            if max_z > 3:
                findings.append(
                    f"Lot size anomaly: max lot {lots.max():.1f} "
                    f"({max_z:.1f}σ above user mean of {mu:.2f})"
                )

    # Single instrument concentration
    top_conc = instrs.value_counts(normalize=True).iloc[0]
    top_instr= instrs.value_counts().index[0]
    if top_conc > 0.80 and n > 10:
        findings.append(
            f"Instrument concentration: {top_conc*100:.0f}% of trades on {top_instr}"
        )

    # Latency arbitrage
    sub30 = dur[dur < 0.5]
    if len(sub30) > 10:
        sub30_pnl = pnl[dur < 0.5]
        win_rate  = (sub30_pnl > 0).mean()
        findings.append(
            f"Latency arbitrage signal: {len(sub30)} sub-30s trades, "
            f"{win_rate*100:.0f}% profitable"
        )

    # High PnL volatility
    if n > 5:
        pnl_vol = pnl.std() / (abs(pnl.mean()) + 1)
        if pnl_vol > 10:
            findings.append(
                f"PnL volatility: std={pnl.std():,.0f}, "
                f"range=[${pnl.min():,.0f}, ${pnl.max():,.0f}]"
            )

    # Consistent small-duration wins (arb)
    if n > 5 and len(sub30) > 5:
        sub30_wins = (pnl[dur < 0.5] > 0).mean()
        if sub30_wins > 0.80:
            findings.append(
                f"Suspicious win rate on fast trades: "
                f"{sub30_wins*100:.0f}% of sub-30s trades profitable"
            )

    return findings


# ── Recommended action mapper ─────────────────────────────────────────────────

def _recommend_action(risk_level: str, findings: list[str]) -> str:
    finding_text = " ".join(findings).lower()

    if risk_level == "CRITICAL":
        if "brute force" in finding_text or "multi-ip" in finding_text:
            return "IMMEDIATE: Freeze account, initiate security review, notify user via secondary channel"
        if "kyc manipulation" in finding_text or "withdrawal" in finding_text:
            return "IMMEDIATE: Block pending withdrawals, escalate to AML team for SAR filing"
        return "IMMEDIATE: Suspend trading, freeze withdrawals, escalate to senior compliance officer"

    if risk_level == "HIGH":
        if "structuring" in finding_text:
            return "URGENT: Flag for AML review within 24h, monitor all transactions"
        if "latency arbitrage" in finding_text or "volume spike" in finding_text:
            return "URGENT: Review trading activity, consider trade reversal if platform abuse confirmed"
        return "URGENT: Enhanced monitoring, request additional KYC documentation within 48h"

    if risk_level == "MEDIUM":
        return "Monitor: Add to watchlist, review manually within 72h"

    return "Log: Flag for routine periodic review"


# ── Main alert builder ────────────────────────────────────────────────────────

def generate_alert(score_result: dict,
                   portal_events: list,
                   trading_events: list) -> dict:
    """
    Args:
        score_result   : output dict from score_user() in async_simulator
        portal_events  : list of raw portal event dicts for this user
        trading_events : list of raw trading event dicts for this user
    Returns:
        structured alert dict
    """
    uid         = score_result["user_id"]
    final_score = score_result["final_score"]
    risk        = _risk_level(final_score)

    portal_findings  = _detect_portal_patterns(portal_events)
    trading_findings = _detect_trading_patterns(trading_events)
    all_findings     = portal_findings + trading_findings

    # Build one-line summary
    if not all_findings:
        summary = (f"Statistical anomaly detected (score={final_score:.3f}). "
                   f"Behavior deviates significantly from peer group.")
    elif len(all_findings) == 1:
        summary = all_findings[0]
    else:
        summary = f"{all_findings[0]}; +{len(all_findings)-1} additional signal(s)"

    action = _recommend_action(risk, all_findings)

    return {
        "alert_id":          f"ALT-{uid}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "user_id":           uid,
        "generated_at":      datetime.utcnow().isoformat(),
        "risk_level":        risk,
        "final_score":       final_score,
        "if_score":          score_result.get("if_score", 0),
        "ae_score":          score_result.get("ae_score", 0),
        "summary":           summary,
        "findings":          all_findings,
        "recommended_action":action,
        "top_features":      score_result.get("top_features", []),
        "portal_event_count":len(portal_events),
        "trading_event_count":len(trading_events),
    }


# ── Batch alert builder (for pre-scored results) ──────────────────────────────

def generate_alerts_from_scores(baseline_scores_path: str = "data/raw/baseline_scores.csv",
                                 portal_path:          str = "data/raw/portal_events.csv",
                                 trading_path:         str = "data/raw/trading_events.csv",
                                 threshold:            float = 0.65,
                                 output_path:          str = "data/raw/alerts.csv") -> pd.DataFrame:
    """Generate batch alerts from pre-computed scores."""
    scores  = pd.read_csv(baseline_scores_path)
    portal  = pd.read_csv(portal_path)
    trading = pd.read_csv(trading_path)

    high_risk = scores[scores["ensemble_score"] >= threshold].copy()
    print(f"[alerts] Generating alerts for {len(high_risk)} high-risk users...")

    alerts = []
    for _, row in high_risk.iterrows():
        uid   = row["user_id"]
        p_evts= portal[portal["user_id"] == uid].to_dict("records")
        t_evts= trading[trading["user_id"] == uid].to_dict("records")

        score_result = {
            "user_id":     uid,
            "final_score": float(row["ensemble_score"]),
            "if_score":    float(row["if_score"]),
            "ae_score":    0.0,
            "top_features": [],
        }
        import json
        if "top_features" in row:
            try:
                score_result["top_features"] = json.loads(row["top_features"])
            except Exception:
                pass

        alerts.append(generate_alert(score_result, p_evts, t_evts))

    df = pd.DataFrame(alerts)
    df.to_csv(output_path, index=False)
    print(f"[alerts] Saved {len(df)} alerts → {output_path}")

    # Pretty print top 5
    print("\n── Top Alerts ───────────────────────────────────────────────")
    for _, a in df.sort_values("final_score", ascending=False).head(5).iterrows():
        print(f"\n  [{a['risk_level']}] {a['user_id']}  score={a['final_score']:.3f}")
        print(f"  Summary: {a['summary']}")
        print(f"  Action : {a['recommended_action']}")

    return df


if __name__ == "__main__":
    generate_alerts_from_scores()
