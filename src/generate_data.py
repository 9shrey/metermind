from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CONSUMER_PROFILES = {
    "Residential": {"load": (1.0, 8.0), "daily": (3.5, 18.0), "pf": (0.82, 0.98)},
    "Commercial": {"load": (5.0, 35.0), "daily": (22.0, 115.0), "pf": (0.78, 0.96)},
    "Industrial": {"load": (30.0, 180.0), "daily": (140.0, 620.0), "pf": (0.72, 0.94)},
}

AREAS = ["Indiranagar", "Whitefield", "Jayanagar", "Koramangala", "Yelahanka", "Peenya", "Electronic City"]


def _consumer_mix(rng: np.random.Generator, n_meters: int) -> np.ndarray:
    return rng.choice(["Residential", "Commercial", "Industrial"], size=n_meters, p=[0.62, 0.27, 0.11])


def generate_smart_meter_data(
    n_meters: int = 540,
    n_days: int = 90,
    tamper_rate: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")
    meter_ids = [f"BESCOM-{idx:05d}" for idx in range(1, n_meters + 1)]
    consumer_types = _consumer_mix(rng, n_meters)
    areas = rng.choice(AREAS, size=n_meters, replace=True)
    feeders = [f"FDR-{area[:3].upper()}-{rng.integers(1, 8):02d}" for area in areas]
    tampered_meters = set(rng.choice(meter_ids, size=int(n_meters * tamper_rate), replace=False))

    rows: list[dict[str, object]] = []

    for idx, meter_id in enumerate(meter_ids):
        consumer_type = consumer_types[idx]
        profile = CONSUMER_PROFILES[consumer_type]
        sanctioned_load = round(rng.uniform(*profile["load"]), 2)
        base_daily = rng.uniform(*profile["daily"]) * rng.normal(1.0, 0.08)
        area_factor = 1 + (AREAS.index(areas[idx]) - len(AREAS) / 2) * 0.015
        feeder_factor = rng.normal(1.0, 0.035)
        is_tampered = meter_id in tampered_meters
        tamper_patterns: set[str] = set()

        if is_tampered:
            tamper_patterns = set(
                rng.choice(
                    ["sudden_drop", "zero_streak", "peer_under", "load_shift", "electrical_irregularity"],
                    size=rng.integers(2, 4),
                    replace=False,
                )
            )
            if "peer_under" in tamper_patterns:
                base_daily *= rng.uniform(0.38, 0.62)

        drop_start = rng.integers(55, 72)
        drop_factor = rng.uniform(0.18, 0.58)
        zero_start = rng.integers(65, 84)
        zero_len = rng.integers(4, 10)

        for day_idx, date in enumerate(dates):
            weekly_factor = 0.9 + 0.16 * np.sin((day_idx % 7) / 7 * 2 * np.pi)
            seasonal_factor = 1 + 0.08 * np.sin(day_idx / n_days * 2 * np.pi)
            random_noise = rng.normal(1.0, 0.16)
            units = base_daily * area_factor * feeder_factor * weekly_factor * seasonal_factor * random_noise

            if consumer_type == "Industrial" and date.weekday() == 6:
                units *= rng.uniform(0.58, 0.78)
            elif consumer_type == "Commercial" and date.weekday() >= 5:
                units *= rng.uniform(0.72, 0.92)

            if "sudden_drop" in tamper_patterns and day_idx >= drop_start:
                units *= drop_factor
            if "zero_streak" in tamper_patterns and zero_start <= day_idx < zero_start + zero_len:
                units = rng.uniform(0.0, 0.4)
            if "load_shift" in tamper_patterns and day_idx >= 60:
                units *= rng.choice([0.28, 0.42, 1.22], p=[0.42, 0.38, 0.20])

            voltage = rng.normal(231.0, 5.5)
            power_factor = rng.uniform(*profile["pf"])
            if "electrical_irregularity" in tamper_patterns and day_idx >= 55 and rng.random() < 0.23:
                voltage = rng.choice([rng.uniform(174, 198), rng.uniform(252, 266)])
                power_factor = rng.uniform(0.50, 0.72)

            rows.append(
                {
                    "meter_id": meter_id,
                    "date": date.date().isoformat(),
                    "consumer_type": consumer_type,
                    "area": areas[idx],
                    "feeder_id": feeders[idx],
                    "sanctioned_load_kw": sanctioned_load,
                    "units_consumed": max(round(float(units), 2), 0.0),
                    "avg_voltage": round(float(voltage), 2),
                    "power_factor": round(float(power_factor), 3),
                    "is_tampered": bool(is_tampered),
                }
            )

    return pd.DataFrame(rows)


def save_sample_data(output_path: str | Path = "data/smart_meter_sample.csv") -> pd.DataFrame:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_smart_meter_data()
    data.to_csv(path, index=False)
    return data


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "data" / "smart_meter_sample.csv"
    df = save_sample_data(output)
    print(f"Saved {len(df):,} rows for {df['meter_id'].nunique():,} meters to {output}")
