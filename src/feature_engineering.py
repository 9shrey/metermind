from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_pct_drop(previous: pd.Series, recent: pd.Series) -> pd.Series:
    return ((previous - recent) / previous.replace(0, np.nan) * 100).fillna(0).clip(lower=0)


def build_meter_features(readings: pd.DataFrame) -> pd.DataFrame:
    df = readings.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["meter_id", "date"])

    latest_date = df["date"].max()
    recent_start = latest_date - pd.Timedelta(days=6)
    previous_start = latest_date - pd.Timedelta(days=36)
    previous_end = latest_date - pd.Timedelta(days=7)

    recent = df[df["date"] >= recent_start]
    previous = df[(df["date"] >= previous_start) & (df["date"] <= previous_end)]

    base = (
        df.groupby("meter_id")
        .agg(
            consumer_type=("consumer_type", "first"),
            area=("area", "first"),
            feeder_id=("feeder_id", "first"),
            sanctioned_load_kw=("sanctioned_load_kw", "first"),
            mean_consumption=("units_consumed", "mean"),
            std_consumption=("units_consumed", "std"),
            zero_consumption_days=("units_consumed", lambda s: int((s <= 0.5).sum())),
            voltage_anomaly_count=("avg_voltage", lambda s: int(((s < 205) | (s > 250)).sum())),
            power_factor_anomaly_count=("power_factor", lambda s: int((s < 0.75).sum())),
            is_tampered=("is_tampered", "max"),
        )
        .reset_index()
    )

    recent_agg = recent.groupby("meter_id").agg(recent_7_day_avg=("units_consumed", "mean")).reset_index()
    previous_agg = previous.groupby("meter_id").agg(previous_30_day_avg=("units_consumed", "mean")).reset_index()

    features = base.merge(recent_agg, on="meter_id", how="left").merge(previous_agg, on="meter_id", how="left")
    features["std_consumption"] = features["std_consumption"].fillna(0)
    features["coefficient_of_variation"] = (
        features["std_consumption"] / features["mean_consumption"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    features["percentage_consumption_drop"] = _safe_pct_drop(
        features["previous_30_day_avg"], features["recent_7_day_avg"]
    )

    peer_baseline = (
        features.groupby(["consumer_type", "area"])["mean_consumption"]
        .transform("median")
        .replace(0, np.nan)
    )
    features["peer_group_deviation"] = ((features["mean_consumption"] - peer_baseline) / peer_baseline * 100).fillna(0)
    features["peer_underconsumption_pct"] = (-features["peer_group_deviation"]).clip(lower=0)

    return features
