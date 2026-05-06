# MeterMind: Behavioral Fingerprinting for Non-Technical Loss Detection

MeterMind is a Streamlit prototype for ranking smart meters by non-technical loss risk. It uses synthetic BESCOM-style smart meter readings, behavioral features, anomaly detection, and rule-based flags to surface meters that may need inspection.

## Problem Statement

Utilities lose revenue when smart meter behavior is distorted by tampering, bypass, faulty wiring, or abnormal load behavior. Field teams need a practical way to prioritize meters before dispatching inspections. MeterMind turns daily meter readings into a risk-ranked investigation queue with plain-language explanations.

## Features

- Generates synthetic smart meter data for 500+ meters across 90 days.
- Includes residential, commercial, and industrial consumers.
- Injects suspicious behavior into roughly 10% of meters.
- Builds consumption, peer-group, voltage, and power-factor features.
- Uses `IsolationForest` plus rule-based risk flags.
- Produces a final 0-100 risk score and Low/Medium/High risk levels.
- Provides explanation flags for every selected meter.
- Includes KPI cards, risk distribution, area-wise high-risk chart, ranking table, and meter-level investigation views.

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly

## How to Run

```bash
cd metermind
pip install -r requirements.txt
python src/generate_data.py
streamlit run app.py
```

The app will also generate `data/smart_meter_sample.csv` automatically if the file is missing.

## Project Structure

```text
metermind/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── smart_meter_sample.csv
├── src/
│   ├── generate_data.py
│   ├── feature_engineering.py
│   ├── anomaly_model.py
│   └── explanations.py
└── screenshots/
```

## Sample Screenshots

Add exported screenshots of the dashboard to the `screenshots/` folder after running the Streamlit app.

Suggested views:

- Dashboard overview with KPI cards and charts.
- Risk ranking table.
- Individual high-risk meter investigation panel.

## Future Scope

- Add hourly interval readings for richer night/day behavioral fingerprints.
- Integrate transformer-level loss balancing and feeder energy audits.
- Add geospatial inspection planning.
- Include model monitoring, drift checks, and feedback from confirmed field inspections.
- Add role-based workflows for analysts, field supervisors, and inspection teams.

## Limitations

- This prototype uses synthetic data only.
- Risk scores are triage indicators, not proof of tampering.
- Operational deployment would require validated smart meter telemetry, meter event logs, field outcomes, and regulatory review.
- The current model is unsupervised and should be calibrated against confirmed non-technical loss cases before production use.
