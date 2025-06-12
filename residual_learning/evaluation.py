#!/usr/bin/env python3
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import gpytorch
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------- BICYCLE PARAMETERS -----------------------
L         = 1.0       # wheel-base [m]
K_DELTA   = 1.38      # steering gain (column → road-wheel)
TAU_DELTA = 0.028     # steering first-order lag [s]
CD        = 3.0e-5    # quadratic drag [s/m]

# ----------------------- HELPERS --------------------------
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

# ----------------------- ENCODER DEFINITIONS -----------------------
class TransformerEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nhead=2, d_hid=128, nlayers=2, dropout=0.1):
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, d_model)
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.outp = torch.nn.Linear(d_model, d_model)
    def forward(self, x):
        h = self.proj(x); h = self.transformer_encoder(h); h = h.mean(dim=1); return self.outp(h)

class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nlayers=2, dropout=0.1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, d_model, nlayers, batch_first=True, dropout=dropout if nlayers > 1 else 0)
    def forward(self, x):
        h_t, _ = self.lstm(x); return h_t[:, -1, :]

class GRUEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nlayers=2, dropout=0.1):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, d_model, nlayers, batch_first=True, dropout=dropout if nlayers > 1 else 0)
    def forward(self, x):
        h_t, _ = self.gru(x); return h_t[:, -1, :]

# ----------------------- SVGP LAYER (Unchanged) -----------------------
class SVGPLayer(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        num_tasks = 1
        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0), batch_shape=torch.Size([num_tasks]))
        variational_strat = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_dist, learn_inducing_locations=True)
        mts = gpytorch.variational.IndependentMultitaskVariationalStrategy(variational_strat, num_tasks=num_tasks)
        super().__init__(mts)
        bs = torch.Size([num_tasks])
        self.mean_module  = gpytorch.means.ConstantMean(batch_shape=bs)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=bs) + gpytorch.kernels.RBFKernel(batch_shape=bs), batch_shape=bs)
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

# ----------------------- MODULAR MODEL DEFINITIONS -----------------------
class ResidualModel(torch.nn.Module):
    def __init__(self, encoder_type, encoder_args, ckpt_args, y_mean, y_std, device):
        super().__init__()
        self.device = device
        self.y_mean = torch.tensor(y_mean, dtype=torch.float32, device=device)
        self.y_std  = torch.tensor(y_std,  dtype=torch.float32, device=device)

        if encoder_type == 'transformer':
            self.encoder = TransformerEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        elif encoder_type == 'lstm':
            self.encoder = LSTMEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        elif encoder_type == 'gru':
            self.encoder = GRUEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        Z = torch.randn(ckpt_args['ind'], ckpt_args['dmod'], device=device)
        self.gp  = SVGPLayer(Z).to(device)
        self.lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1).to(device)

    def forward(self, x):
        z = self.encoder(x); dist = self.gp(z); pred_norm = self.lik(dist).mean
        return pred_norm * self.y_std + self.y_mean

class DeterministicResidualModel(torch.nn.Module):
    def __init__(self, encoder_type, encoder_args, ckpt_args, y_mean, y_std, device):
        super().__init__()
        self.device = device
        self.y_mean = torch.tensor(y_mean, dtype=torch.float32, device=device)
        self.y_std  = torch.tensor(y_std,  dtype=torch.float32, device=device)

        if encoder_type == 'transformer':
            self.encoder = TransformerEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        elif encoder_type == 'lstm':
            self.encoder = LSTMEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        elif encoder_type == 'gru':
            self.encoder = GRUEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
            
        self.regressor = torch.nn.Linear(ckpt_args['dmod'], 1).to(device)

    def forward(self, x):
        z = self.encoder(x); pred_norm = self.regressor(z)
        return pred_norm * self.y_std + self.y_mean

# ----------------------- SIMULATION & EVALUATION --------------------------
def run_base_simulation(meas, ctrl, dt):
    print("Running base simulation (kinematic model only)...")
    N = len(meas); sim_states = []; state = meas[0].copy()
    for k in tqdm(range(N), desc="Base Sim"):
        sim_states.append(state)
        if k < N - 1: state = simulate_step(state, ctrl[k], dt)
    return {'pos_x': np.array([s['pos_x'] for s in sim_states]), 'pos_y': np.array([s['pos_y'] for s in sim_states]), 'yaw': np.array([s['yaw'] for s in sim_states]), 'speed': np.array([s['speed'] for s in sim_states])}

def run_corrected_simulation(model, meas, ctrl, H, dt, device):
    print("Running corrected simulation (kinematic + residual model)...")
    N = len(meas); model.eval(); sim_states = []; state = meas[0].copy()
    history_features = torch.zeros((H, 6), device=device)
    for k in tqdm(range(N), desc="Corrected Sim"):
        sim_states.append(state)
        if k >= N - 1: continue
        current_features = feat_tensor(state, ctrl[k], device)
        history_features = torch.cat([history_features[1:], current_features.unsqueeze(0)], dim=0)
        rz_residual = 0.0
        if k >= H:
            with torch.no_grad():
                rz_residual = model(history_features.unsqueeze(0)).item()
        next_state_base = simulate_step(state, ctrl[k], dt)
        corrected_rz = next_state_base['r_z'] + rz_residual
        next_yaw_corr = wrap_to_pi(state['yaw'] + corrected_rz * dt)
        state = next_state_base.copy()
        state['yaw'] = next_yaw_corr; state['r_z'] = corrected_rz
    return {'pos_x': np.array([s['pos_x'] for s in sim_states]), 'pos_y': np.array([s['pos_y'] for s in sim_states]), 'yaw': np.array([s['yaw'] for s in sim_states]), 'speed': np.array([s['speed'] for s in sim_states])}

def calculate_rmse(real_yaw, real_speed, real_pos_x, real_pos_y, sim_dict):
    yaw_err = np.array([wrap_to_pi(ry - sy) for ry, sy in zip(real_yaw, sim_dict['yaw'])])
    yaw_rmse = np.sqrt(np.mean(yaw_err**2))
    speed_rmse = np.sqrt(np.mean((real_speed - sim_dict['speed'])**2))
    pos_err = np.sqrt((real_pos_x - sim_dict['pos_x'])**2 + (real_pos_y - sim_dict['pos_y'])**2)
    pos_rmse = pos_err[-1]
    return pos_rmse, speed_rmse, yaw_rmse

def plot_results(times, real, sim_base, sim_corr, model_info_str):
    print("Generating plots...")
    base_rmse = calculate_rmse(real['yaw'], real['speed'], real['pos_x'], real['pos_y'], sim_base)
    corr_rmse = calculate_rmse(real['yaw'], real['speed'], real['pos_x'], real['pos_y'], sim_corr)
    print(f"\n--- RMSE Results ---")
    print(f"Base Sim:           Pos. Error (final)={base_rmse[0]:.3f}m, Speed RMSE={base_rmse[1]:.3f}m/s, Yaw RMSE={base_rmse[2]:.3f}rad")
    print(f"Corrected Sim ({model_info_str}): Pos. Error (final)={corr_rmse[0]:.3f}m, Speed RMSE={corr_rmse[1]:.3f}m/s, Yaw RMSE={corr_rmse[2]:.3f}rad\n")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Real vs. Simulated Vehicle Dynamics (Correction Model: {model_info_str})', fontsize=16)
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.plot(real['pos_x'], real['pos_y'], 'k-', lw=2, label='Ground Truth')
    ax1.plot(sim_base['pos_x'], sim_base['pos_y'], 'g--', label=f'Base Sim (Final Pos Err: {base_rmse[0]:.2f} m)')
    ax1.plot(sim_corr['pos_x'], sim_corr['pos_y'], 'r-', lw=2.0, label=f'Corrected Sim (Final Pos Err: {corr_rmse[0]:.2f} m)')
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]"); ax1.set_title("Trajectory Comparison"); ax1.legend(); ax1.grid(True); ax1.axis('equal')
    ax2 = axes[1, 0]
    ax2.plot(times, real['speed'], 'k-', label='Ground Truth'); ax2.plot(times, sim_base['speed'], 'g--', label=f'Base Sim (RMSE: {base_rmse[1]:.3f})'); ax2.plot(times, sim_corr['speed'], 'r-', label=f'Corrected Sim (RMSE: {corr_rmse[1]:.3f})')
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("speed [m/s]"); ax2.set_title("Speed Comparison"); ax2.legend(); ax2.grid(True)
    ax3 = axes[1, 1]
    ax3.plot(times, [wrap_to_pi(y) for y in real['yaw']], 'k-', label='Ground Truth'); ax3.plot(times, [wrap_to_pi(y) for y in sim_base['yaw']], 'g--', label=f'Base Sim (RMSE: {base_rmse[2]:.3f})'); ax3.plot(times, [wrap_to_pi(y) for y in sim_corr['yaw']], 'r-', label=f'Corrected Sim (RMSE: {corr_rmse[2]:.3f})')
    ax3.set_xlabel("time [s]"); ax3.set_ylabel("yaw [rad]"); ax3.set_title("Yaw Comparison"); ax3.legend(); ax3.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_name = f"evaluation_results_{model_info_str.lower().replace(' ', '_')}.png"
    plt.savefig(save_name)
    print(f"Saved plot to {save_name}")
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a trained residual dynamics model.")
    p.add_argument("--csv", required=True, help="Path to the test data CSV file.")
    p.add_argument("--model", required=True, help="Path to the saved model .pt file.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (cuda or cpu).")
    args = p.parse_args()

    print(f"Loading model from {args.model} onto {args.device}...")
    ckpt = torch.load(args.model, map_location=args.device)
    model_args = ckpt['args']

    # --- DYNAMICALLY LOAD THE CORRECT MODEL AND ENCODER ---
    encoder_type = ckpt.get('encoder_type', 'transformer') # Default to transformer if not specified
    print(f"Detected Encoder Type: {encoder_type.upper()}")
    
    encoder_args = {}
    if encoder_type == 'transformer':
        encoder_args = {'nhead': model_args.get('tf_nhead', 2), 'd_hid': model_args.get('tf_d_hid', 128), 'nlayers': model_args.get('tf_nlayers', 2), 'dropout': model_args.get('tf_dropout', 0.1)}
    else: # lstm or gru
        encoder_args = {'nlayers': model_args.get('rnn_nlayers', 2), 'dropout': model_args.get('rnn_dropout', 0.1)}

    if 'gp' in ckpt:
        print("Detected Probabilistic (SVGP) model.")
        model_variant = 'SVGP'
        model = ResidualModel(
            encoder_type=encoder_type, encoder_args=encoder_args,
            ckpt_args=model_args, y_mean=ckpt['y_mean'], y_std=ckpt['y_std'], device=args.device
        )
        model.encoder.load_state_dict(ckpt['encoder'])
        model.gp.load_state_dict(ckpt['gp'])
        model.lik.load_state_dict(ckpt['lik'])
    elif 'regressor' in ckpt:
        print("Detected Deterministic (Linear) model.")
        model_variant = 'Linear'
        model = DeterministicResidualModel(
            encoder_type=encoder_type, encoder_args=encoder_args,
            ckpt_args=model_args, y_mean=ckpt['y_mean'], y_std=ckpt['y_std'], device=args.device
        )
        model.encoder.load_state_dict(ckpt['encoder'])
        model.regressor.load_state_dict(ckpt['regressor'])
    else:
        raise ValueError("Could not determine model variant from checkpoint. Missing 'gp' or 'regressor' keys.")

    print("Model loaded successfully.")
    
    # --- Data Loading (Unchanged) ---
    print(f"Loading test data from {args.csv}...")
    df = pd.read_csv(Path(args.csv)); times = df["time"].values.astype(np.float32)
    real = {'pos_x': df["pos_x"].values.astype(np.float32), 'pos_y': df["pos_y"].values.astype(np.float32), 'speed': df["speed"].values.astype(np.float32), 'yaw': df["yaw"].values.astype(np.float32)}
    acc_cmd   = df["acceleration"].values.astype(np.float32); steer_cmd = np.deg2rad(df["steer_deg"].values) * K_DELTA
    N = len(times); dt = float(np.mean(np.diff(times))); r_z_meas = np.zeros(N, dtype=np.float32)
    r_z_meas[1:] = np.diff(real['yaw']) / dt
    meas, ctrl = [], []
    for k in range(N):
        meas.append({'pos_x': real['pos_x'][k], 'pos_y': real['pos_y'][k], 'yaw': real['yaw'][k], 'speed': real['speed'][k], 'r_z': r_z_meas[k], 'a_x': acc_cmd[k], 'delta_prev': steer_cmd[k-1] if k > 0 else steer_cmd[0]})
        ctrl.append({'acc': acc_cmd[k], 'delta_cmd': steer_cmd[k]})

    # --- RUN SIMULATIONS AND PLOT ---
    sim_base = run_base_simulation(meas, ctrl, dt)
    sim_corr = run_corrected_simulation(model, meas, ctrl, model_args['hist'], dt, args.device)

    model_info_str = f"{encoder_type.upper()} {model_variant}"
    plot_results(times, real, sim_base, sim_corr, model_info_str)