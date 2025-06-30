#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py - Main executable for training residual dynamics models.

This script orchestrates the entire training pipeline:
1. Parses command-line arguments.
2. Loads and splits data in a session-aware manner.
3. Builds the training dataset and dataloader.
4. Initializes the specified model (SVGP/Linear + Encoder).
5. Calls the training loop from the trainer module.
6. Plots the final results.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import our custom modules from the 'src' directory
from src.config import K_DELTA
from src.data_loader import build_data, WindowDataset
from src.models import ResidualModel_SVGP, ResidualModel_Linear
from src.trainer import train_model
from src.utils import plot_curves


def convert_df_to_lists(df):
    """Helper to convert a dataframe into the list-of-dicts format."""
    meas, ctrl = [], []
    for k in range(len(df)):
        meas.append({
            'pos_x': df['pos_x'].iloc[k], 'pos_y': df['pos_y'].iloc[k],
            'yaw': df['yaw'].iloc[k], 'speed': df['speed'].iloc[k],
            'a_x': df['acceleration'].iloc[k],
            'r_z': df['r_z'].iloc[k],  # This line requires the 'r_z' column to exist
            'delta_prev': np.deg2rad(df['steer_deg'].iloc[k-1] if k > 0 else df['steer_deg'].iloc[0]) * K_DELTA
        })
        ctrl.append({
            'acc': df['acceleration'].iloc[k],
            'delta_cmd': np.deg2rad(df['steer_deg'].iloc[k]) * K_DELTA
        })
    return meas, ctrl


def main():
    """Main orchestration function."""
    p = argparse.ArgumentParser(description="Train a modular residual dynamics model, optimizing for positional error.")
    p.add_argument("--csv", required=True, help="Path to the AGGREGATED training data CSV (must contain 'session_id').")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--val_sessions", type=int, default=2, help="Number of complete sessions to hold out for validation.")
    p.add_argument("--eval_freq", type=int, default=10, help="Run full validation simulation every N epochs.")
    p.add_argument("--hist", type=int, default=10, help="History window size.")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    p.add_argument("--wd", type=float, default=1e-4, help="Weight decay.")
    p.add_argument("--save_dir", default="./saved_models/", help="Directory to save the best model checkpoints.")
    p.add_argument("--model_type", type=str, required=True, choices=["svgp", "linear"], help="Type of prediction model head.")
    p.add_argument("--encoder", type=str, required=True, choices=["transformer", "lstm", "gru", "mobilevit"], help="Type of sequence encoder.")
    p.add_argument("--dmod", type=int, default=64, help="Model dimension (feature size).")
    p.add_argument("--ind", type=int, default=150, help="Number of inducing points for SVGP.")
    p.add_argument("--k_roll", type=int, default=5, help="Number of steps for rollout loss.")
    p.add_argument("--rz_weight", type=float, default=1.0, help="Weight for the rollout loss term.")
    p.add_argument("--tf_nhead", type=int, default=4, help="Number of heads for Transformer.")
    p.add_argument("--tf_d_hid", type=int, default=128, help="Hidden dimension for Transformer's FFN.")
    p.add_argument("--tf_nlayers", type=int, default=2, help="Number of layers for Transformer.")
    p.add_argument("--tf_dropout", type=float, default=0.1, help="Dropout for Transformer.")
    p.add_argument("--rnn_nlayers", type=int, default=2, help="Number of layers for LSTM/GRU.")
    p.add_argument("--rnn_dropout", type=float, default=0.1, help="Dropout for LSTM/GRU.")
    args = p.parse_args()

    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    args.save = Path(args.save_dir) / "residual_model.pt"
    print(f"Device: {device}, Model: {args.model_type.upper()}, Encoder: {args.encoder.upper()}")

    # --- 2. Load and Split Data (Session-Aware) ---
    df_all = pd.read_csv(Path(args.csv))
    assert 'session_id' in df_all.columns, "CSV must contain a 'session_id' column."
    
    all_session_ids = df_all['session_id'].unique()
    if len(all_session_ids) <= args.val_sessions:
        raise ValueError(f"Not enough sessions to hold out {args.val_sessions} for validation.")
    
    val_session_ids = all_session_ids[-args.val_sessions:]
    train_session_ids = all_session_ids[:-args.val_sessions]
    
    df_train = df_all[df_all['session_id'].isin(train_session_ids)].copy()
    df_val = df_all[df_all['session_id'] == val_session_ids[0]].copy().reset_index(drop=True)
    
    print(f"Data split: {len(train_session_ids)} sessions for training, {len(val_session_ids)} for validation.")
    print(f"Using session {val_session_ids[0]} for validation simulation checks.")

    # --- 3. Prepare Data Structures ---
    dt = float(np.mean(df_all.groupby('session_id')['time'].diff().dropna()))

    # ==============================================================================
    # --- FIX: CALCULATE YAW RATE (r_z) BEFORE USING IT ---
    print("Calculating yaw rate (r_z) for each session...")
    # Use np.gradient for a more stable derivative, calculated per session.
    # This adds the 'r_z' column to the DataFrames.
    df_train['r_z'] = df_train.groupby('session_id')['yaw'].transform(lambda x: np.gradient(np.unwrap(x), dt))
    df_val['r_z'] = df_val.groupby('session_id')['yaw'].transform(lambda x: np.gradient(np.unwrap(x), dt))
    # ==============================================================================
    
    # Now that 'r_z' column exists, this function call will succeed
    train_meas, train_ctrl = convert_df_to_lists(df_train)
    val_meas, val_ctrl = convert_df_to_lists(df_val)

    # --- 4. Build Windowed Dataset for Training ---
    # The build_data function needs the raw DataFrame to group it internally
    X_train, Y_train = build_data(df_train, args.hist, dt)
    
    y_mean = Y_train.mean(dim=0).numpy(); y_std = Y_train.std(dim=0).numpy() + 1e-8
    Y_norm = (Y_train - torch.from_numpy(y_mean)) / torch.from_numpy(y_std)
    
    train_ds = WindowDataset(X_train, Y_norm)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    
    print("\n" + "="*50); print(f"Values needed for evaluation:\n--y_mean {y_mean[0]:.8f}\n--y_std  {y_std[0]:.8f}"); print("="*50 + "\n")

    # --- 5. Initialize Model ---
    encoder_args = {'nlayers': args.rnn_nlayers, 'dropout': args.rnn_dropout} if args.encoder != 'transformer' else {'nhead': args.tf_nhead, 'd_hid': args.tf_d_hid, 'nlayers': args.tf_nlayers, 'dropout': args.tf_dropout}
    
    model = None
    if args.model_type == 'svgp':
        model = ResidualModel_SVGP(args.encoder, encoder_args, y_mean, y_std, args.dmod, args.ind, args.lr, args.wd, device).to(device)
        model.mll.num_data = len(train_ds)
    elif args.model_type == 'linear':
        model = ResidualModel_Linear(args.encoder, encoder_args, y_mean, y_std, args.dmod, args.lr, args.wd, device).to(device)

    # --- 6. Train Model ---
    train_history, val_history, val_epochs = train_model(
        model=model,
        train_loader=train_loader,
        val_meas=val_meas,
        val_ctrl=val_ctrl,
        train_meas=train_meas,
        train_ctrl=train_ctrl,
        dt=dt,
        device=device,
        epochs=args.epochs,
        args=args
    )

    print("\nTraining complete. The best model (based on positional error) has been saved.")

    # --- 7. Plot Results ---
    if train_history and val_history:
        plot_curves(train_history, val_history, val_epochs, args.model_type, args.encoder)

if __name__ == "__main__":
    main()