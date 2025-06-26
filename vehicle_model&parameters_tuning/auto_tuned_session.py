#!/usr/bin/env python3
"""
Robust auto-tuning script for a kinematic bicycle model:
- MODIFIED: Now session-aware to handle concatenated data files correctly.
- Tunes wheel-base (L), steering gain (K_delta), steering lag (tau_delta), and drag (C_d).
- Handles edge cases: avoids zero-lag overflows, clamps steering to safe range.
Usage:
    python auto_tuned_bicycle_model.py training_data_all_sessions.csv [--plot]
"""
import argparse
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Attempt to import SciPy
try:
    from scipy.optimize import differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import matplotlib.pyplot as plt

# Constants
MIN_TAU = 1e-3
MAX_STEER_RAD = math.pi/2 - 1e-3

def load_log(csv_path: Path):
    df = pd.read_csv(csv_path)
    required = {"time", "pos_x", "pos_y", "yaw", "speed", "steer_deg", "acceleration", "session_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {required - set(df.columns)}")
    df = df.astype({c: float for c in required if c != 'session_id'})
    dt = float(np.round(df.groupby('session_id')['time'].diff().mean(), 6))
    return df, dt

def simulate(df, dt, L, K_delta, tau_delta, C_d):
    # This function is correct as is, as it simulates a single continuous trajectory.
    L = max(L, 0.1)
    tau = max(tau_delta, MIN_TAU)
    steer_cmd = np.deg2rad(df["steer_deg"].values) * K_delta
    accel_cmd = df["acceleration"].values
    N = len(df)
    sx, sy, syaw, sv = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    x, y, yaw, v = df["pos_x"].iloc[0], df["pos_y"].iloc[0], df["yaw"].iloc[0], df["speed"].iloc[0]
    delta = steer_cmd[0]
    for i in range(N):
        sx[i], sy[i], syaw[i], sv[i] = x, y, yaw, v
        delta += (steer_cmd[i] - delta) * (dt / tau)
        delta = max(min(delta, MAX_STEER_RAD), -MAX_STEER_RAD)
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += (v / L) * math.tan(delta) * dt
        v += (accel_cmd[i] - C_d * v * v) * dt
    return sx, sy, syaw, sv

def loss(params, df_all_sessions, dt, weights):
    # --- MODIFICATION: Make the loss function session-aware ---
    total_loss = 0.0
    num_sessions = 0
    try:
        L, K_delta, tau_delta, C_d = params
        
        # Group by session and run a simulation for each one
        for session_id, df_session in df_all_sessions.groupby('session_id'):
            if len(df_session) < 2: continue # Skip very short sessions

            sx, sy, syaw, sv = simulate(df_session, dt, L, K_delta, tau_delta, C_d)

            # np.unwrap is crucial for yaw error calculation across boundaries
            yaw_err = np.abs(np.unwrap(df_session["yaw"].values) - np.unwrap(syaw))
            
            # Check for NaN/inf in simulation output, which indicates instability
            if np.isnan(sx).any() or np.isinf(sx).any():
                return 1e6 # Return a large penalty if simulation fails

            pos_err = np.hypot(df_session["pos_x"] - sx, df_session["pos_y"] - sy)
            speed_err = np.abs(df_session["speed"] - sv)
            
            session_loss = (weights["pos"] * pos_err.mean() +
                            weights["yaw"] * yaw_err.mean() +
                            weights["speed"] * speed_err.mean())
            
            total_loss += session_loss
            num_sessions += 1

        if num_sessions == 0: return 1e6
        return total_loss / num_sessions # Return the average loss over all sessions

    except Exception:
        return 1e6 # Return a large penalty if any error occurs

def calibrate(df, dt, bounds, weights, maxiter=100, seed=1):
    if SCIPY_AVAILABLE:
        print("Calibrating with differential_evolution (SciPy)...")
        result = differential_evolution(
            loss, bounds=list(bounds.values()), args=(df, dt, weights),
            seed=seed, maxiter=maxiter, popsize=15, polish=True, disp=True
        )
        return result.x, result.fun
    else:
        print("Warning: SciPy not found. Falling back to basic random search.")
        rng = np.random.default_rng(seed)
        best, best_loss = None, 1e9
        for _ in tqdm(range(maxiter * 15 * 10), desc="Random Search"): # More iterations for fallback
            guess = [rng.uniform(l,h) for (l,h) in bounds.values()]
            l = loss(guess, df, dt, weights)
            if l < best_loss:
                best, best_loss = guess, l
        return best, best_loss

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__)
    )
    p.add_argument("csv", type=Path, help="Path to the concatenated CSV file with a 'session_id' column.")
    p.add_argument("--plot", action="store_true", help="Show a plot of the first session's trajectory.")
    args = p.parse_args()

    df, dt = load_log(args.csv)
    
    bounds = {"L": (0.5, 4.0), "K_delta": (0.5, 2.0), "tau_delta": (MIN_TAU, 0.5), "C_d": (0.0, 0.1)}
    weights = {"pos": 1.0, "yaw": 2.0, "speed": 1.0}
    
    print(f"Calibrating on {df['session_id'].nunique()} sessions (this may take a few minutes)...")
    params, score = calibrate(df, dt, bounds, weights)
    L, K_delta, tau_delta, C_d = params
    
    print("\n" + "="*50)
    print(textwrap.dedent(f"""
    Tuned parameters:
      L          = {L:.4f} m
      K_delta    = {K_delta:.4f}
      tau_delta  = {tau_delta:.4f} s
      C_d        = {C_d:.6f} s/m
    Final Average Loss = {score:.4f}
    """))
    
    # --- MODIFICATION: Calculate final RMSEs across all sessions ---
    all_pos_err, all_yaw_err, all_speed_err = [], [], []
    for _, session_df in df.groupby('session_id'):
        if len(session_df) < 2: continue
        sx, sy, syaw, sv = simulate(session_df, dt, *params)
        all_pos_err.extend(np.hypot(session_df['pos_x']-sx, session_df['pos_y']-sy))
        all_yaw_err.extend(np.unwrap(session_df['yaw']) - np.unwrap(syaw))
        all_speed_err.extend(session_df['speed'] - sv)

    print(textwrap.dedent(f"""
    Overall RMSE Metrics:
    RMSE position = {rmse(np.array(all_pos_err), 0):.3f} m
    RMSE yaw      = {rmse(np.array(all_yaw_err), 0):.3f} rad
    RMSE speed    = {rmse(np.array(all_speed_err), 0):.3f} m/s
    """))
    print("="*50)

    if args.plot:
        print("\nPlotting trajectory for the first session...")
        first_session_df = df[df['session_id'] == df['session_id'].unique()[0]]
        sx, sy, syaw, sv = simulate(first_session_df, dt, *params)
        
        plt.figure(figsize=(8, 8))
        plt.plot(first_session_df['pos_x'], first_session_df['pos_y'], label='Real Trajectory (First Session)')
        plt.plot(sx, sy, '--', label='Simulated Trajectory')
        plt.axis('equal'); plt.grid(True); plt.legend()
        plt.title('Real vs. Simulated (Calibrated on All Sessions)')
        plt.xlabel('x [m]'); plt.ylabel('y [m]')
        
        output_filename = "tuned_result.png"
        plt.savefig(output_filename)
        print(f"Plot saved to {output_filename}")
        plt.show()

if __name__ == "__main__":
    if not SCIPY_AVAILABLE:
        print("Warning: SciPy is not installed. `pip install scipy` for much better optimization results.")
    main()