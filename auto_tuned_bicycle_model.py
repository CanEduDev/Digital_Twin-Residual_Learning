#!/usr/bin/env python3
"""
Robust auto-tuning script for a kinematic bicycle model:
- Tunes wheel-base (L), steering gain (K_delta), steering lag (tau_delta), and drag (C_d).
- Handles edge cases: avoids zero-lag overflows, clamps steering to safe range.
Usage:
    python auto_tuned_bicycle_model_v2.py data.csv [--plot]
"""
import argparse
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# Attempt to import SciPy
try:
    from scipy.optimize import differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import matplotlib.pyplot as plt

# Constants
MIN_TAU = 1e-3              # minimal steering lag to avoid inf gain
MAX_STEER_RAD = math.pi/2 - 1e-3  # clamp steer angle before tan()

def load_log(csv_path: Path):
    df = pd.read_csv(csv_path)
    required = {"time","pos_x","pos_y","yaw","speed","steer_deg","acceleration"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {required - set(df.columns)}")
    df = df.astype({c:float for c in required})
    dt = float(np.round(df["time"].diff().dropna().iloc[0],6))
    if dt <= 0: dt = float(df["time"].diff().mean())
    return df, dt

def simulate(df, dt, L, K_delta, tau_delta, C_d):
    # enforce safe ranges
    L = max(L, 0.1)
    tau = max(tau_delta, MIN_TAU)
    steer_cmd = np.deg2rad(df["steer_deg"].values) * K_delta
    accel_cmd = df["acceleration"].values

    N = len(df)
    sx = np.zeros(N); sy = np.zeros(N)
    syaw = np.zeros(N); sv = np.zeros(N)

    x, y, yaw, v = df["pos_x"].iloc[0], df["pos_y"].iloc[0], df["yaw"].iloc[0], df["speed"].iloc[0]
    delta = steer_cmd[0]

    for i in range(N):
        sx[i], sy[i], syaw[i], sv[i] = x, y, yaw, v

        # first-order steering lag
        delta += (steer_cmd[i] - delta)*(dt / tau)
        # clamp to avoid tan overflow
        delta = max(min(delta, MAX_STEER_RAD), -MAX_STEER_RAD)

        # kinematic bicycle step
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += (v / L) * math.tan(delta) * dt
        v += (accel_cmd[i] - C_d * v * v) * dt

    return sx, sy, syaw, sv

def loss(params, df, dt, weights):
    try:
        L, K_delta, tau_delta, C_d = params
        sx, sy, syaw, sv = simulate(df, dt, L, K_delta, tau_delta, C_d)
        pos_err = np.hypot(df["pos_x"]-sx, df["pos_y"]-sy)
        yaw_err = np.abs(np.unwrap(df["yaw"].values)-np.unwrap(syaw))
        speed_err = np.abs(df["speed"]-sv)
        return (weights["pos"]*pos_err.mean()
                + weights["yaw"]*yaw_err.mean()
                + weights["speed"]*speed_err.mean())
    except Exception:
        return 1e6

def calibrate(df, dt, bounds, weights, maxiter=100, seed=1):
    if SCIPY_AVAILABLE:
        result = differential_evolution(
            loss,
            bounds=list(bounds.values()),
            args=(df,dt,weights),
            seed=seed,
            maxiter=maxiter,
            popsize=10,
            polish=True
        )
        return result.x, result.fun
    # fallback random search
    rng = np.random.default_rng(seed)
    best, best_loss = None, 1e9
    for _ in range(maxiter):
        guess = [rng.uniform(l,h) for (l,h) in bounds.values()]
        l = loss(guess, df, dt, weights)
        if l < best_loss:
            best, best_loss = guess, l
    return best, best_loss

def rmse(a,b):
    return float(np.sqrt(np.mean((a-b)**2)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    df, dt = load_log(args.csv)
    bounds = {
        "L":(1.0,5.0),
        "K_delta":(0.5,2.0),
        "tau_delta":(MIN_TAU,0.5),
        "C_d":(0.0,0.1),
    }
    weights = {"pos":1.0,"yaw":0.5,"speed":2.0}
    print("Calibrating (this may take a minute)...")
    params, score = calibrate(df, dt, bounds, weights)
    L, K_delta, tau_delta, C_d = params
    print(textwrap.dedent(f"""
    Tuned parameters:
      L          = {L:.3f} m
      K_delta    = {K_delta:.3f}
      tau_delta  = {tau_delta:.3f} s
      C_d        = {C_d:.5f} s/m
    Loss = {score:.3f}
    """))
    # final metrics
    sx, sy, syaw, sv = simulate(df, dt, *params)
    print(textwrap.dedent(f"""
    RMSE position = {rmse(np.hypot(df['pos_x']-sx,df['pos_y']-sy),0):.3f} m
    RMSE yaw      = {rmse(np.unwrap(df['yaw']), np.unwrap(syaw)):.3f} rad
    RMSE speed    = {rmse(df['speed'], sv):.3f} m/s
    """))
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        plt.plot(df['pos_x'], df['pos_y'], label='Real')
        plt.plot(sx, sy, '--', label='Sim')
        plt.axis('equal'); plt.grid(True); plt.legend()
        plt.title('Real vs Simulated (calibrated)')
        plt.xlabel('x [m]'); plt.ylabel('y [m]')
        plt.show()
        plt.savefig('tuned_result.png')

if __name__=="__main__":
    main()
