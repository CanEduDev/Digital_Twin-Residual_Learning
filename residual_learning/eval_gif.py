#!/usr/bin/env python3
"""
long_term_evaluation_v3.py - A comprehensive evaluation script.

This script evaluates a trained model's long-term prediction fidelity by:
1. Running a full closed-loop simulation to get the primary trajectory.
2. Generating a set of ROLLING MULTI-STEP PREDICTIONS, where each prediction
   is RESET to the ground truth state at each time step.
3. Analyzing the error as a function of the prediction horizon.
4. Saving the horizon error analysis to a CSV file.
5. Generating a trajectory plot, a horizon error plot, and a GIF of the simulation.
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
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# --- BICYCLE PARAMETERS & HELPERS (Unchanged) ---
L, K_DELTA, TAU_DELTA, CD = 1.0, 1.38, 0.028, 3.0e-5
def wrap_to_pi(angle: float) -> float: return math.atan2(math.sin(angle), math.cos(angle))
def simulate_step(state: dict, control: dict, dt: float) -> dict:
    delta = state['delta_prev'] + (dt / TAU_DELTA) * (control['delta_cmd'] - state['delta_prev']); x = state['pos_x'] + state['speed'] * math.cos(state['yaw']) * dt
    y = state['pos_y'] + state['speed'] * math.sin(state['yaw']) * dt; psi = wrap_to_pi(state['yaw'] + (state['speed'] / L) * math.tan(delta) * dt)
    v = state['speed'] + (control['acc'] - CD * state['speed']**2) * dt; r_z = wrap_to_pi(psi - state['yaw']) / dt; a_x = (v - state['speed']) / dt
    return {'pos_x': x, 'pos_y': y, 'yaw': psi, 'speed': v, 'delta_prev': delta, 'r_z': r_z, 'a_x': a_x}
def feat_tensor(state: dict, control: dict, device) -> torch.Tensor:
    return torch.tensor([state['a_x'], state['speed'], math.sin(state['yaw']), math.cos(state['yaw']), control['acc'], control['delta_cmd']], dtype=torch.float32, device=device)

# --- MODEL AND ENCODER DEFINITIONS (Unchanged) ---
class TransformerEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nhead=2, d_hid=128, nlayers=2, dropout=0.1):
        super().__init__(); self.proj = torch.nn.Linear(input_dim, d_model); encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True); self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers); self.final_norm = torch.nn.LayerNorm(d_model); self.outp = torch.nn.Linear(d_model, d_model)
    def forward(self, x): h = self.proj(x); h = self.transformer_encoder(h); h = h.mean(dim=1); h = self.outp(h); return self.final_norm(h)
class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nlayers=2, dropout=0.1):
        super().__init__(); self.lstm = torch.nn.LSTM(input_dim, d_model, nlayers, batch_first=True, dropout=dropout if nlayers > 1 else 0); self.final_norm = torch.nn.LayerNorm(d_model)
    def forward(self, x): h_t, _ = self.lstm(x); return self.final_norm(h_t[:, -1, :])
class GRUEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nlayers=2, dropout=0.1):
        super().__init__(); self.gru = torch.nn.GRU(input_dim, d_model, nlayers, batch_first=True, dropout=dropout if nlayers > 1 else 0); self.final_norm = torch.nn.LayerNorm(d_model)
    def forward(self, x): h_t, _ = self.gru(x); return self.final_norm(h_t[:, -1, :])
class MobileViTEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nhead=2, d_hid=128, nlayers=2, dropout=0.1):
        super().__init__(); self.conv_block = torch.nn.Sequential(torch.nn.Conv1d(input_dim, d_model, 3, 1, 1), torch.nn.BatchNorm1d(d_model), torch.nn.SiLU(), torch.nn.Conv1d(d_model, d_model, 3, 1, 1), torch.nn.BatchNorm1d(d_model), torch.nn.SiLU()); encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True); self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers); self.final_norm = torch.nn.LayerNorm(d_model); self.outp = torch.nn.Linear(d_model, d_model)
    def forward(self, x): x_conv = x.permute(0, 2, 1); local_features = self.conv_block(x_conv); local_features = local_features.permute(0, 2, 1); global_features = self.transformer_encoder(local_features); fused_features = global_features + local_features; h = fused_features.mean(dim=1); return self.final_norm(self.outp(h))
class SVGPLayer(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        num_tasks = 1; variational_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0), batch_shape=torch.Size([num_tasks])); variational_strat = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_dist, learn_inducing_locations=True); mts = gpytorch.variational.IndependentMultitaskVariationalStrategy(variational_strat, num_tasks=num_tasks); super().__init__(mts); bs = torch.Size([num_tasks]); self.mean_module  = gpytorch.means.ConstantMean(batch_shape=bs); self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=bs) + gpytorch.kernels.RBFKernel(batch_shape=bs), batch_shape=bs)
    def forward(self, x): return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))
class ResidualModel(torch.nn.Module):
    def __init__(self, encoder_type, encoder_args, ckpt_args, y_mean, y_std, device):
        super().__init__(); self.device = device; self.y_mean = torch.tensor(y_mean, dtype=torch.float32, device=device); self.y_std  = torch.tensor(y_std,  dtype=torch.float32, device=device)
        if encoder_type == 'transformer': self.encoder = TransformerEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        elif encoder_type == 'lstm': self.encoder = LSTMEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        elif encoder_type == 'gru': self.encoder = GRUEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        elif encoder_type == 'mobilevit': self.encoder = MobileViTEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        else: raise ValueError(f"Unknown encoder type: {encoder_type}")
        Z = torch.randn(ckpt_args['ind'], ckpt_args['dmod'], device=device); self.gp  = SVGPLayer(Z).to(device); self.lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1).to(device)
    def forward(self, x): z = self.encoder(x); dist = self.gp(z); pred_norm = self.lik(dist).mean; return pred_norm * self.y_std + self.y_mean

# --- SIMULATION AND ANALYSIS FUNCTIONS (Unchanged) ---
def run_closed_loop_simulation(model, meas, ctrl, H, dt, device):
    print("Running main closed-loop simulation..."); model.eval()
    N = len(meas); sim_states = []; state = meas[0].copy(); history_features = torch.zeros((H, 6), device=device)
    for k in tqdm(range(N), desc="Closed-Loop Sim"):
        sim_states.append(state.copy());
        if k >= N - 1: continue
        current_features = feat_tensor(state, ctrl[k], device); history_features = torch.cat([history_features[1:], current_features.unsqueeze(0)], dim=0)
        rz_residual = model(history_features.unsqueeze(0)).item() if k >= H else 0.0
        next_state_base = simulate_step(state, ctrl[k], dt); corrected_rz = next_state_base['r_z'] + rz_residual
        next_yaw_corr = wrap_to_pi(state['yaw'] + corrected_rz * dt); state = next_state_base.copy()
        state['yaw'] = next_yaw_corr; state['r_z'] = corrected_rz
    return {key: np.array([s[key] for s in sim_states]) for key in sim_states[0]}

def generate_rolling_future_predictions(model, meas, ctrl, H, dt, device, prediction_horizon):
    print("Generating rolling future predictions (from ground truth)..."); model.eval()
    N = len(meas); future_predictions = []
    gt_history_buffer = torch.zeros((N, H, 6), device=device, dtype=torch.float32)
    for i in range(N):
        for j in range(H):
            k = i - (H - 1) + j
            if k >= 0: gt_history_buffer[i, j] = feat_tensor(meas[k], ctrl[k], device)
    for k in tqdm(range(N), desc="Rolling Predictions"):
        if k >= N - prediction_horizon: future_predictions.append([]); continue
        current_horizon_preds = []; future_state = meas[k].copy(); future_history = gt_history_buffer[k].clone()
        with torch.no_grad():
            for j in range(prediction_horizon):
                step_idx = k + j
                rz_residual = model(future_history.unsqueeze(0)).item()
                next_future_state_base = simulate_step(future_state, ctrl[step_idx], dt)
                corrected_rz = next_future_state_base['r_z'] + rz_residual; next_yaw_corr = wrap_to_pi(future_state['yaw'] + corrected_rz * dt)
                future_state = next_future_state_base.copy(); future_state['yaw'] = next_yaw_corr; future_state['r_z'] = corrected_rz
                current_horizon_preds.append(future_state)
                next_features = feat_tensor(future_state, ctrl[step_idx], device); future_history = torch.cat([future_history[1:], next_features.unsqueeze(0)], dim=0)
        future_predictions.append(current_horizon_preds)
    return future_predictions

def analyze_horizon_errors(real_data, future_predictions, horizon):
    print("Analyzing horizon prediction errors..."); horizon_errors = []
    real_pos = np.stack([real_data['pos_x'], real_data['pos_y']], axis=1)
    for h in range(1, horizon + 1):
        step_errors_sq = []
        for k in range(len(future_predictions) - h):
            if len(future_predictions[k]) >= h:
                pred_state = future_predictions[k][h - 1]; pred_pos = np.array([pred_state['pos_x'], pred_state['pos_y']])
                true_pos = real_pos[k + h]; error_sq = np.sum((pred_pos - true_pos)**2); step_errors_sq.append(error_sq)
        if step_errors_sq: horizon_errors.append(np.sqrt(np.mean(step_errors_sq)))
        else: horizon_errors.append(np.nan)
    return np.array(horizon_errors)

def create_gif_from_frames(times, real, sim_main_path, future_predictions, model_info_str, horizon, gif_name, dt):
    temp_dir = "temp_gif_frames";
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True); frame_files = []
    N = len(times); frame_step = max(1, N // 150)
    print("Generating frames for GIF...");
    for k in tqdm(range(0, N, frame_step), desc="Generating Frames"):
        fig, ax = plt.subplots(figsize=(12, 8)); ax.plot(real['pos_x'], real['pos_y'], 'k-', lw=3, alpha=0.8, label='Ground Truth')
        ax.plot(sim_main_path['pos_x'], sim_main_path['pos_y'], 'r-', lw=2.5, alpha=0.5, label='Full Sim Path (for context)')
        ax.plot(real['pos_x'][k], real['pos_y'][k], 'o', color='black', ms=12, label='Current Real Position')
        if k < len(future_predictions) and future_predictions[k]:
            start_x, start_y = real['pos_x'][k], real['pos_y'][k]
            pred_x = [start_x] + [p['pos_x'] for p in future_predictions[k]]; pred_y = [start_y] + [p['pos_y'] for p in future_predictions[k]]
            # --- MODIFICATION: Make the prediction line thinner ---
            ax.plot(pred_x, pred_y, 'b--o', lw=1.5, ms=4, label=f'{horizon}-Step Future Prediction')
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title(f'Rolling Multi-Step Prediction ({model_info_str})\nTime: {times[k]:.2f}s', fontsize=16)
        ax.legend(loc='best'); ax.grid(True); ax.axis('equal'); frame_path = os.path.join(temp_dir, f"frame_{k:05d}.png")
        plt.savefig(frame_path, dpi=90); plt.close(fig); frame_files.append(frame_path)
    print(f"Stitching {len(frame_files)} frames into {gif_name}...");
    with imageio.get_writer(gif_name, mode='I', duration=1000 * dt * frame_step, loop=0) as writer:
        for filename in tqdm(frame_files, desc="Stitching GIF"): writer.append_data(imageio.v2.imread(filename))
    shutil.rmtree(temp_dir); print(f"✅ GIF saved successfully to {gif_name}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate a GIF and horizon error analysis of a residual dynamics model.")
    p.add_argument("--csv", required=True); p.add_argument("--model", required=True)
    p.add_argument("--horizon", type=int, default=50, help="Number of future steps to predict and analyze (e.g., 50 steps).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-gif", action='store_true', help="Skip generating the GIF to save time.")
    args = p.parse_args()

    print(f"Loading model from {args.model} onto {args.device}..."); ckpt = torch.load(args.model, map_location=args.device)
    model_args = ckpt['args']
    encoder_type = ckpt.get('encoder_type', 'transformer'); print(f"Detected Encoder Type: {encoder_type.upper()}")
    encoder_args = {'nhead': model_args.get('tf_nhead', 2), 'd_hid': model_args.get('tf_d_hid', 128), 'nlayers': model_args.get('tf_nlayers', 2)} if encoder_type in ['transformer', 'mobilevit'] else {'nlayers': model_args.get('rnn_nlayers', 2)}
    
    model = ResidualModel(encoder_type, encoder_args, model_args, ckpt['y_mean'], ckpt['y_std'], args.device)
    model.encoder.load_state_dict(ckpt['encoder']); model.gp.load_state_dict(ckpt['gp']); model.lik.load_state_dict(ckpt['lik'])
    print("Model loaded successfully.")
    
    print(f"Loading test data from {args.csv}..."); df = pd.read_csv(Path(args.csv)); times = df["time"].values
    real = {'pos_x': df["pos_x"].values, 'pos_y': df["pos_y"].values, 'speed': df["speed"].values, 'yaw': df["yaw"].values}
    acc_cmd = df["acceleration"].values; steer_cmd = np.deg2rad(df["steer_deg"].values) * K_DELTA
    N = len(times); dt = float(np.mean(np.diff(times)));
    df['r_z'] = np.gradient(np.unwrap(df['yaw']), dt)
    meas, ctrl = [], []
    for k in range(N):
        meas.append({'pos_x': df['pos_x'].iloc[k], 'pos_y': df['pos_y'].iloc[k], 'yaw': df['yaw'].iloc[k], 'speed': df['speed'].iloc[k], 'a_x': acc_cmd[k], 'r_z': df['r_z'].iloc[k], 'delta_prev': steer_cmd[k-1] if k > 0 else steer_cmd[0]})
        ctrl.append({'acc': acc_cmd[k], 'delta_cmd': steer_cmd[k]})

    # --- Run Simulations and Analysis ---
    sim_main_path = run_closed_loop_simulation(model, meas, ctrl, model_args['hist'], dt, args.device)
    future_predictions = generate_rolling_future_predictions(model, meas, ctrl, model_args['hist'], dt, args.device, args.horizon)
    horizon_errors = analyze_horizon_errors(real, future_predictions, args.horizon)
    
    model_info_str = f"{encoder_type.upper()}_SVGP"
    
    # --- NEW: Save Horizon Errors to CSV ---
    horizon_steps = np.arange(1, args.horizon + 1); horizon_ms = horizon_steps * dt * 1000
    error_df = pd.DataFrame({
        'horizon_steps': horizon_steps,
        'horizon_ms': horizon_ms,
        'rmse_positional_error_m': horizon_errors
    })
    error_csv_path = f"horizon_error_summary_{model_info_str.lower()}.csv"
    error_df.to_csv(error_csv_path, index=False)
    print(f"Saved horizon error data to {error_csv_path}")

    # --- Generate Plots ---
    plt.style.use('seaborn-v0_8-whitegrid')
    # Plot 1: Error vs. Prediction Horizon
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(horizon_ms, horizon_errors, '-o', label=f'{model_info_str}')
    ax.set_xlabel(f'Prediction Horizon (ms)  [1 step = {dt*1000:.1f} ms]'); ax.set_ylabel('Average Positional Error (m) [RMSE]')
    ax.set_title('Long-Term Prediction Fidelity'); ax.grid(True); ax.legend()
    plt.tight_layout(); plt.savefig(f"evaluation_horizon_error_{model_info_str.lower()}.png"); plt.show()

    # Generate GIF (optional)
    if not args.no_gif:
        gif_name = f"evaluation_rolling_prediction_{model_info_str.lower()}_h{args.horizon}.gif"
        create_gif_from_frames(times, real, sim_main_path, future_predictions, model_info_str, args.horizon, gif_name, dt)