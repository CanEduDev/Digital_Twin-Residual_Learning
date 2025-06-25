#!/usr/bin/env python3
"""
evaluation_multistep_gif.py

This script evaluates a trained residual dynamics model by performing multi-step-ahead
predictions at each time step and visualizes the results as a GIF.

At each time step 'k' of the simulation, it:
1.  Performs a closed-loop update to determine the state at 'k+1'. This forms the main simulated trajectory.
2.  From the state at 'k', performs an open-loop prediction for 'k+1' through 'k+5' to visualize
    the model's future-state prediction capability.
3.  Generates a GIF comparing the ground truth, the main simulated path, and the rolling future predictions.
"""
import os
import shutil
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import gpytorch
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt

# ----------------------- BICYCLE PARAMETERS (Unchanged) -----------------------
L         = 1.0       # wheel-base [m]
K_DELTA   = 1.38      # steering gain (column → road-wheel)
TAU_DELTA = 0.028     # steering first-order lag [s]
CD        = 3.0e-5    # quadratic drag [s/m]

# ----------------------- HELPERS (Unchanged) --------------------------
def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

def simulate_step(state: dict, control: dict, dt: float) -> dict:
    delta = state['delta_prev'] + (dt / TAU_DELTA) * (control['delta_cmd'] - state['delta_prev'])
    x     = state['pos_x'] + state['speed'] * math.cos(state['yaw']) * dt
    y     = state['pos_y'] + state['speed'] * math.sin(state['yaw']) * dt
    psi   = wrap_to_pi(state['yaw'] + (state['speed'] / L) * math.tan(delta) * dt)
    v     = state['speed'] + (control['acc'] - CD * state['speed']**2) * dt
    r_z   = (psi - state['yaw']) / dt
    a_x   = (v - state['speed']) / dt
    return {'pos_x': x, 'pos_y': y, 'yaw': psi, 'speed': v, 'delta_prev': delta, 'r_z': r_z, 'a_x': a_x}

def feat_tensor(state: dict, control: dict, device) -> torch.Tensor:
    return torch.tensor([
        state['a_x'], state['speed'],
        math.sin(state['yaw']), math.cos(state['yaw']),
        control['acc'], control['delta_cmd']
    ], dtype=torch.float32, device=device)

# ----------------------- ENCODER DEFINITIONS (Must match training script) -----------------------
class TransformerEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nhead=2, d_hid=128, nlayers=2, dropout=0.1):
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, d_model)
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.final_norm = torch.nn.LayerNorm(d_model)
        self.outp = torch.nn.Linear(d_model, d_model)
    def forward(self, x):
        h = self.proj(x); h = self.transformer_encoder(h); h = h.mean(dim=1); h = self.outp(h); return self.final_norm(h)

class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nlayers=2, dropout=0.1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, d_model, nlayers, batch_first=True, dropout=dropout if nlayers > 1 else 0)
        self.final_norm = torch.nn.LayerNorm(d_model)
    def forward(self, x):
        h_t, _ = self.lstm(x); return self.final_norm(h_t[:, -1, :])

class GRUEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nlayers=2, dropout=0.1):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, d_model, nlayers, batch_first=True, dropout=dropout if nlayers > 1 else 0)
        self.final_norm = torch.nn.LayerNorm(d_model)
    def forward(self, x):
        h_t, _ = self.gru(x); return self.final_norm(h_t[:, -1, :])

# ----------------------- SVGP LAYER (Unchanged) -----------------------
class SVGPLayer(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        num_tasks = 1; variational_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0), batch_shape=torch.Size([num_tasks]))
        variational_strat = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_dist, learn_inducing_locations=True)
        mts = gpytorch.variational.IndependentMultitaskVariationalStrategy(variational_strat, num_tasks=num_tasks); super().__init__(mts)
        bs = torch.Size([num_tasks]); self.mean_module  = gpytorch.means.ConstantMean(batch_shape=bs)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=bs) + gpytorch.kernels.RBFKernel(batch_shape=bs), batch_shape=bs)
    def forward(self, x): return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

# ----------------------- MODEL DEFINITIONS (Must match training script) -----------------------
class ResidualModel(torch.nn.Module): # SVGP Model
    def __init__(self, encoder_type, encoder_args, ckpt_args, y_mean, y_std, device):
        super().__init__(); self.device = device
        self.y_mean = torch.tensor(y_mean, dtype=torch.float32, device=device); self.y_std  = torch.tensor(y_std,  dtype=torch.float32, device=device)
        if encoder_type == 'transformer': self.encoder = TransformerEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        elif encoder_type == 'lstm': self.encoder = LSTMEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        elif encoder_type == 'gru': self.encoder = GRUEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        else: raise ValueError(f"Unknown encoder type: {encoder_type}")
        Z = torch.randn(ckpt_args['ind'], ckpt_args['dmod'], device=device); self.gp  = SVGPLayer(Z).to(device); self.lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1).to(device)
    def forward(self, x):
        z = self.encoder(x); dist = self.gp(z); pred_norm = self.lik(dist).mean; return pred_norm * self.y_std + self.y_mean

class DeterministicResidualModel(torch.nn.Module):
    def __init__(self, encoder_type, encoder_args, ckpt_args, y_mean, y_std, device):
        super().__init__(); self.device = device
        self.y_mean = torch.tensor(y_mean, dtype=torch.float32, device=device); self.y_std  = torch.tensor(y_std,  dtype=torch.float32, device=device)
        if encoder_type == 'transformer': self.encoder = TransformerEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        elif encoder_type == 'lstm': self.encoder = LSTMEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        elif encoder_type == 'gru': self.encoder = GRUEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        else: raise ValueError(f"Unknown encoder type: {encoder_type}")
        self.regressor = torch.nn.Linear(ckpt_args['dmod'], 1).to(device)
    def forward(self, x):
        z = self.encoder(x); pred_norm = self.regressor(z); return pred_norm * self.y_std + self.y_mean

# ----------------------- NEW: MULTI-STEP SIMULATION --------------------------
def run_multistep_simulation(model, meas, ctrl, H, dt, device, prediction_horizon):
    """
    Runs a simulation that performs multi-step ahead predictions at each time step.
    Returns the main closed-loop trajectory and the list of future predictions.
    """
    print("Running multi-step corrected simulation...")
    N = len(meas)
    model.eval()

    # To store the main simulated path (closed-loop)
    sim_states = []
    # To store the 5-step future predictions made at each time step
    future_predictions = []

    # Initialize state and history
    state = meas[0].copy()
    history_features = torch.zeros((H, 6), device=device)

    for k in tqdm(range(N), desc="Multi-Step Sim"):
        sim_states.append(state)
        if k >= N - 1:
            future_predictions.append([]) # Append empty list for the last step
            continue

        # --- 1. Update history with the current state for the next prediction ---
        current_features = feat_tensor(state, ctrl[k], device)
        history_features = torch.cat([history_features[1:], current_features.unsqueeze(0)], dim=0)

        # --- 2. Perform multi-step future prediction (open-loop from current state) ---
        current_horizon_preds = []
        future_state = state.copy()
        future_history = history_features.clone()

        with torch.no_grad():
            for j in range(prediction_horizon):
                step_idx = k + j
                if step_idx >= N - 1: break # Stop if we run out of control inputs

                # Predict residual from the current future history
                rz_residual = model(future_history.unsqueeze(0)).item()

                # Simulate one step forward
                next_future_state_base = simulate_step(future_state, ctrl[step_idx], dt)
                corrected_rz = next_future_state_base['r_z'] + rz_residual
                next_yaw_corr = wrap_to_pi(future_state['yaw'] + corrected_rz * dt)

                # Update state for the *next* iteration of this inner loop
                future_state = next_future_state_base.copy()
                future_state['yaw'] = next_yaw_corr
                future_state['r_z'] = corrected_rz
                current_horizon_preds.append(future_state)

                # Update history for the *next* iteration of this inner loop
                next_features = feat_tensor(future_state, ctrl[step_idx], device)
                future_history = torch.cat([future_history[1:], next_features.unsqueeze(0)], dim=0)
        
        future_predictions.append(current_horizon_preds)

        # --- 3. Perform single-step prediction for main trajectory (closed-loop) ---
        if k >= H:
             with torch.no_grad():
                # This uses the history_features updated at the start of the loop
                rz_residual_main = model(history_features.unsqueeze(0)).item()
        else:
            rz_residual_main = 0.0

        next_state_base = simulate_step(state, ctrl[k], dt)
        corrected_rz_main = next_state_base['r_z'] + rz_residual_main
        next_yaw_corr_main = wrap_to_pi(state['yaw'] + corrected_rz_main * dt)

        # This state is used for the next iteration of the main loop (k+1)
        state = next_state_base.copy()
        state['yaw'] = next_yaw_corr_main
        state['r_z'] = corrected_rz_main

    # Convert lists of dicts to dicts of arrays for easier plotting
    sim_main_path = {key: np.array([s[key] for s in sim_states]) for key in sim_states[0]}
    return sim_main_path, future_predictions


# ----------------------- NEW: GIF CREATION --------------------------
def create_gif_from_frames(times, real, sim_main_path, future_predictions, model_info_str, horizon, gif_name):
    """Generates frames and stitches them into a GIF."""
    temp_dir = "temp_gif_frames"
    os.makedirs(temp_dir, exist_ok=True)
    frame_files = []
    
    N = len(times)
    frame_step = 5 # Create a frame every 5 time steps to keep GIF fast
    
    print("Generating frames for GIF...")
    for k in tqdm(range(0, N, frame_step), desc="Generating Frames"):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the full ground truth path
        ax.plot(real['pos_x'], real['pos_y'], 'k-', lw=2, label='Ground Truth')
        
        # Plot the main simulated path up to the current time k
        ax.plot(sim_main_path['pos_x'][:k+1], sim_main_path['pos_y'][:k+1], 'r-', lw=2.5, label='Simulated Path (Closed-Loop)')
        
        # Plot the future prediction made at time k
        if k < len(future_predictions) and future_predictions[k]:
            preds = future_predictions[k]
            pred_x = [p['pos_x'] for p in preds]
            pred_y = [p['pos_y'] for p in preds]
            ax.plot(pred_x, pred_y, 'b--o', lw=1.5, ms=4, label=f'{horizon}-Step Future Prediction')

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f'Multi-Step Prediction (Model: {model_info_str})\nTime: {times[k]:.2f}s', fontsize=16)
        ax.legend(loc='upper left')
        ax.grid(True)
        ax.axis('equal')
        
        frame_path = os.path.join(temp_dir, f"frame_{k:05d}.png")
        plt.savefig(frame_path, dpi=100)
        plt.close(fig)
        frame_files.append(frame_path)

    print(f"Stitching {len(frame_files)} frames into {gif_name}...")
    with imageio.get_writer(gif_name, mode='I', duration=1000 * dt * frame_step, loop=0) as writer:
        for filename in tqdm(frame_files, desc="Stitching GIF"):
            image = imageio.imread(filename)
            writer.append_data(image)
    
    print("Cleaning up temporary frames...")
    shutil.rmtree(temp_dir)
    print(f"✅ GIF saved successfully to {gif_name}")


# ----------------------- MAIN EXECUTION -----------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a residual model with multi-step prediction and generate a GIF.")
    p.add_argument("--csv", required=True, help="Path to the test data CSV file.")
    p.add_argument("--model", required=True, help="Path to the saved model .pt file.")
    p.add_argument("--horizon", type=int, default=5, help="Number of steps to predict into the future.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (cuda or cpu).")
    args = p.parse_args()

    print(f"Loading model from {args.model} onto {args.device}...")
    # Add weights_only=False for compatibility with older PyTorch versions and NumPy arrays in checkpoint
    ckpt = torch.load(args.model, map_location=args.device, weights_only=False)
    model_args = ckpt['args']
    
    # --- Dynamically load the correct model and encoder ---
    encoder_type = ckpt.get('encoder_type', 'transformer')
    print(f"Detected Encoder Type: {encoder_type.upper()}")
    
    encoder_args = {}
    if encoder_type == 'transformer':
        encoder_args = {'nhead': model_args.get('tf_nhead', 2), 'd_hid': model_args.get('tf_d_hid', 128), 'nlayers': model_args.get('tf_nlayers', 2), 'dropout': model_args.get('tf_dropout', 0.1)}
    else: # lstm or gru
        encoder_args = {'nlayers': model_args.get('rnn_nlayers', 2), 'dropout': model_args.get('rnn_dropout', 0.1)}

    if 'gp' in ckpt:
        print("Detected Probabilistic (SVGP) model.")
        model_variant = 'SVGP'
        model = ResidualModel(encoder_type, encoder_args, model_args, ckpt['y_mean'], ckpt['y_std'], args.device)
        model.encoder.load_state_dict(ckpt['encoder'])
        model.gp.load_state_dict(ckpt['gp'])
        model.lik.load_state_dict(ckpt['lik'])
    elif 'regressor' in ckpt:
        print("Detected Deterministic (Linear) model.")
        model_variant = 'Linear'
        model = DeterministicResidualModel(encoder_type, encoder_args, model_args, ckpt['y_mean'], ckpt['y_std'], args.device)
        model.encoder.load_state_dict(ckpt['encoder'])
        model.regressor.load_state_dict(ckpt['regressor'])
    else:
        raise ValueError("Could not determine model variant from checkpoint. Missing 'gp' or 'regressor' keys.")
    print("Model loaded successfully.")
    
    # --- Data Loading ---
    print(f"Loading test data from {args.csv}...")
    df = pd.read_csv(Path(args.csv))
    times = df["time"].values.astype(np.float32)
    real = {'pos_x': df["pos_x"].values, 'pos_y': df["pos_y"].values, 'speed': df["speed"].values, 'yaw': df["yaw"].values}
    acc_cmd = df["acceleration"].values.astype(np.float32)
    steer_cmd = np.deg2rad(df["steer_deg"].values) * K_DELTA
    N = len(times); dt = float(np.mean(np.diff(times)))
    r_z_meas = np.zeros(N, dtype=np.float32); r_z_meas[1:] = np.diff(real['yaw']) / dt
    meas, ctrl = [], []
    for k in range(N):
        meas.append({'pos_x': real['pos_x'][k], 'pos_y': real['pos_y'][k], 'yaw': real['yaw'][k], 'speed': real['speed'][k], 'r_z': r_z_meas[k], 'a_x': acc_cmd[k], 'delta_prev': steer_cmd[k-1] if k > 0 else steer_cmd[0]})
        ctrl.append({'acc': acc_cmd[k], 'delta_cmd': steer_cmd[k]})

    # --- RUN SIMULATION AND GENERATE GIF ---
    sim_main_path, future_predictions = run_multistep_simulation(
        model, meas, ctrl, model_args['hist'], dt, args.device, args.horizon
    )

    model_info_str = f"{encoder_type.upper()}_{model_variant}"
    gif_name = f"evaluation_{model_info_str.lower()}_h{args.horizon}.gif"
    
    create_gif_from_frames(times, real, sim_main_path, future_predictions, model_info_str, args.horizon, gif_name)

