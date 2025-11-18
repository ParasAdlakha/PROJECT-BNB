import pandas as pd
import numpy as np
from scipy.stats import linregress


def compute_kpis(df: pd.DataFrame) -> list:
    """
    Computes critical KPIs from the raw flight data, focusing on degradation signs.
    """
    kpis = []

    # --- 1. Command-to-Position Lag (Core Actuator Anomaly) ---
    # Calculate the instantaneous error (lag)
    df['lag'] = df['cmd_deg'] - df['pos_deg']
    mean_lag = df['lag'].mean()
    std_lag = df['lag'].std()

    kpis.append({
        "signal_name": "cmd_pos_lag_avg",
        "mean_value": round(mean_lag, 3),
        "std_dev": round(std_lag, 3),
        "metric_type": "Actuator Performance",
        "description": f"Average lag between commanded and measured angle. Target is near zero. Mean: {round(mean_lag, 3)} degrees."
    })

    # --- 2. Hydraulic Pressure Trend (Internal Leakage/Wear Indicator) ---
    # Use linear regression to find the slope of pressure over time_step (cycles)
    try:
        slope, intercept, r_value, p_value, std_err = linregress(df['time_step'], df['hyd_pressure_psi'])
        trend_context = "Positive slope suggests increasing pressure required for operation, potentially due to internal leakage or wear." if slope > 0.001 else "Trend is stable."
    except ValueError:
        slope = 0.0 # Handle case where data might be too short

    kpis.append({
        "signal_name": "hyd_pressure_trend",
        "mean_value": round(df['hyd_pressure_psi'].mean(), 1),
        "trend_slope": round(slope * 1000, 4), # Scale slope for better readability
        "metric_type": "System Health",
        "description": f"Pressure trend per 1000 time steps/cycles. Slope: {round(slope * 1000, 4)} psi. {trend_context}"
    })

    # --- 3. Actuator Temperature Max/Mean (Friction/Overload Indicator) ---
    kpis.append({
        "signal_name": "actuator_temp_C_max",
        "mean_value": round(df['actuator_temp_C'].mean(), 1),
        "max_value": round(df['actuator_temp_C'].max(), 1),
        "metric_type": "Thermal Health",
        "description": f"Max observed temperature. Max: {round(df['actuator_temp_C'].max(), 1)} C."
    })

    return kpis

def analyze_run(file_data: bytes, run_id: str) -> list:
    """Main entry point for data analysis."""
    # Convert bytes to DataFrame
    from io import BytesIO
    df = pd.read_csv(BytesIO(file_data))

    # Basic data validation/cleanup (optional, but good practice)
    df.columns = df.columns.str.lower().str.replace('[^a-z0-9_]', '', regex=True)

    # Check for required columns
    required_cols = ['time_step', 'cmd_deg', 'pos_deg', 'hyd_pressure_psi', 'actuator_temp_c']
    if not all(col in df.columns for col in required_cols):
        # Try to rename common case variation
        df = df.rename(columns={"actuator_temp_c": "actuator_temp_C"})
        if not all(col in df.columns for col in required_cols):
             raise ValueError(f"Missing required columns in CSV. Found: {df.columns.tolist()}")

    # Compute and return KPIs
    kpis = compute_kpis(df)

    return kpis