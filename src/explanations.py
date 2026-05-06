from __future__ import annotations

import pandas as pd


def explain_meter(row: pd.Series) -> list[str]:
    explanations: list[str] = []

    if row["percentage_consumption_drop"] >= 25:
        explanations.append(
            f"Consumption dropped {row['percentage_consumption_drop']:.0f}% compared to previous 30-day baseline."
        )
    if row["zero_consumption_days"] >= 3:
        explanations.append(f"Detected {int(row['zero_consumption_days'])} zero-consumption days in the observation window.")
    if row["peer_underconsumption_pct"] >= 25:
        explanations.append("Usage is significantly lower than similar consumers in the same area.")
    if row["voltage_anomaly_count"] >= 2:
        explanations.append("Voltage readings show irregular behavior.")
    if row["power_factor_anomaly_count"] >= 3:
        explanations.append("Power factor readings indicate abnormal load behavior.")
    if row["coefficient_of_variation"] >= 0.65:
        explanations.append("Consumption pattern is unusually volatile for this meter.")

    if not explanations:
        explanations.append("No major non-technical loss indicators were detected for this meter.")

    return explanations


def attach_explanations(scored: pd.DataFrame) -> pd.DataFrame:
    result = scored.copy()
    result["explanations"] = result.apply(explain_meter, axis=1)
    return result
