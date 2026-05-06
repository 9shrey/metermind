from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


MODEL_FEATURES = [
    "mean_consumption",
    "std_consumption",
    "recent_7_day_avg",
    "previous_30_day_avg",
    "percentage_consumption_drop",
    "zero_consumption_days",
    "coefficient_of_variation",
    "peer_underconsumption_pct",
    "voltage_anomaly_count",
    "power_factor_anomaly_count",
]


def score_meters(features: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    scored = features.copy()
    model_input = scored[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaled = RobustScaler().fit_transform(model_input)

    model = IsolationForest(n_estimators=250, contamination=0.10, random_state=random_state)
    model.fit(scaled)

    raw_scores = -model.decision_function(scaled)
    model_risk = 100 * (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)

    rule_risk = (
        np.clip(scored["percentage_consumption_drop"] / 85 * 32, 0, 32)
        + np.clip(scored["zero_consumption_days"] / 12 * 20, 0, 20)
        + np.clip(scored["peer_underconsumption_pct"] / 70 * 18, 0, 18)
        + np.clip(scored["voltage_anomaly_count"] / 10 * 15, 0, 15)
        + np.clip(scored["power_factor_anomaly_count"] / 14 * 15, 0, 15)
    )

    scored["model_anomaly_score"] = model_risk.round(2)
    scored["rule_risk_score"] = rule_risk.round(2)
    scored["risk_score"] = np.clip((0.58 * model_risk) + (0.42 * rule_risk), 0, 100).round(1)
    scored["risk_level"] = pd.cut(
        scored["risk_score"],
        bins=[-0.01, 39, 69, 100],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    return scored.sort_values("risk_score", ascending=False).reset_index(drop=True)
