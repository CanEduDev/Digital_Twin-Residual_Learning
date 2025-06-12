#!/usr/bin/env python3
"""
train_multi_encoder_svgp.py - MODULAR & OPTIMIZED FOR SPEED

Train an SVGP residual model for YAW RATE ONLY with a choice of sequence encoders
(Transformer, LSTM, or GRU) and genuine multi-step rollout loss.
"""
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import gpytorch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

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

def simulate_step_torch(states: dict, controls: dict, dt: float) -> dict:
    delta = states['delta_prev'] + (dt / TAU_DELTA) * (controls['delta_cmd'] - states['delta_prev'])
    x     = states['pos_x'] + states['speed'] * torch.cos(states['yaw']) * dt
    y     = states['pos_y'] + states['speed'] * torch.sin(states['yaw']) * dt
    unwrapped_psi = states['yaw'] + (states['speed'] / L) * torch.tan(delta) * dt
    psi   = torch.atan2(torch.sin(unwrapped_psi), torch.cos(unwrapped_psi))
    v     = states['speed'] + (controls['acc'] - CD * states['speed']**2) * dt
    r_z   = (psi - states['yaw']) / dt
    a_x   = (v - states['speed']) / dt
    return {'pos_x': x, 'pos_y': y, 'yaw': psi, 'speed': v, 'delta_prev': delta, 'r_z': r_z, 'a_x': a_x}

def feat_tensor_torch(states: dict, controls: dict) -> torch.Tensor:
    return torch.stack([
        states['a_x'], states['speed'],
        torch.sin(states['yaw']), torch.cos(states['yaw']),
        controls['acc'], controls['delta_cmd']
    ], dim=1)

# ----------------------- DATASET HELPERS --------------------------
class WindowDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X; self.Y = Y
    def __len__(self): return self.X.size(0)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx], idx

# ----------------------- ENCODER DEFINITIONS -----------------------
class TransformerEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nhead=2, d_hid=128, nlayers=2, dropout=0.1):
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, d_model)
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.outp = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        h = self.proj(x)
        h = self.transformer_encoder(h)
        # Use the features from the last time step
        h = h.mean(dim=1)
        return self.outp(h)

class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nlayers=2, dropout=0.1):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=d_model,
            num_layers=nlayers,
            batch_first=True,
            dropout=dropout if nlayers > 1 else 0
        )
    def forward(self, x):
        # We only care about the last output, not the hidden states
        # h_t shape: (batch_size, sequence_length, hidden_size)
        h_t, _ = self.lstm(x)
        # Return the output of the last time step
        return h_t[:, -1, :]

class GRUEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nlayers=2, dropout=0.1):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=d_model,
            num_layers=nlayers,
            batch_first=True,
            dropout=dropout if nlayers > 1 else 0
        )
    def forward(self, x):
        h_t, _ = self.gru(x)
        return h_t[:, -1, :]

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

# ----------------------- MODULAR RESIDUAL MODEL -----------------------
class ResidualModel(torch.nn.Module):
    def __init__(self, encoder_type, encoder_args, y_mean, y_std, d_model, n_inducing, lr, weight_decay, device):
        super().__init__()
        self.device = device
        self.y_mean = torch.tensor(y_mean, dtype=torch.float32, device=device)
        self.y_std  = torch.tensor(y_std,  dtype=torch.float32, device=device)
        
        # --- SELECT AND BUILD THE CHOSEN ENCODER ---
        if encoder_type == 'transformer':
            self.encoder = TransformerEncoder(input_dim=6, d_model=d_model, **encoder_args).to(device)
        elif encoder_type == 'lstm':
            self.encoder = LSTMEncoder(input_dim=6, d_model=d_model, **encoder_args).to(device)
        elif encoder_type == 'gru':
            self.encoder = GRUEncoder(input_dim=6, d_model=d_model, **encoder_args).to(device)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        Z = torch.randn(n_inducing, d_model, device=device)
        self.gp  = SVGPLayer(Z).to(device)
        self.lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1).to(device)
        self.mll = gpytorch.mlls.VariationalELBO(self.lik, self.gp, num_data=1)
        self.opt = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            {'params': self.gp.parameters(),      'lr': lr, 'weight_decay': weight_decay},
            {'params': self.lik.parameters(),     'lr': lr}
        ])

    def forward(self, x):
        z    = self.encoder(x)
        dist = self.gp(z)
        pred_norm = self.lik(dist).mean
        return pred_norm * self.y_std + self.y_mean

# ----------------------- BUILD DATA (Unchanged) -----------------------
def build_data(meas, ctrl, H, dt):
    N = len(meas); M = N - (H + 1)
    X = np.zeros((M, H, 6), np.float32); Y = np.zeros((M, 1), np.float32)
    print("Building one-step residual training data for yaw rate (r_z)...")
    for i in tqdm(range(M)):
        for j in range(H):
            k = i + j
            X[i, j] = [meas[k]['a_x'], meas[k]['speed'], math.sin(meas[k]['yaw']), math.cos(meas[k]['yaw']), ctrl[k]['acc'], ctrl[k]['delta_cmd']]
        state0 = meas[i+H-1].copy(); control0 = ctrl[i+H-1].copy()
        next_state = simulate_step(state0, control0, dt)
        Y[i, 0] = meas[i+H]['r_z'] - next_state['r_z']
    return torch.from_numpy(X), torch.from_numpy(Y)

# ----------------------- TRAIN + ROLLOUT (Unchanged) -----------------------
def train_rollout(model, train_loader, val_loader, meas, ctrl, H, dt, device, epochs, args, patience=10):
    scheduler = CosineAnnealingWarmRestarts(model.opt, T_0=20, T_mult=2)
    N = len(meas); K_roll = args.k_roll

    for ep in range(1, epochs+1):
        model.train(); tot_elbo=0.0; tot_mse=0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [{args.encoder.upper()}]")
        for bx, by, idx in pbar:
            bx, by = bx.to(device), by.to(device)
            model.opt.zero_grad()
            dist_norm = model.gp(model.encoder(bx))
            elbo_loss = -model.mll(dist_norm, by)
            B = bx.size(0); hists = bx.clone()
            i0s = idx.numpy(); initial_indices = i0s + H - 1
            batch_states = {
                'pos_x': torch.tensor([meas[i]['pos_x'] for i in initial_indices], device=device, dtype=torch.float32),
                'pos_y': torch.tensor([meas[i]['pos_y'] for i in initial_indices], device=device, dtype=torch.float32),
                'yaw':   torch.tensor([meas[i]['yaw'] for i in initial_indices], device=device, dtype=torch.float32),
                'speed': torch.tensor([meas[i]['speed'] for i in initial_indices], device=device, dtype=torch.float32),
                'delta_prev': torch.tensor([meas[i]['delta_prev'] for i in initial_indices], device=device, dtype=torch.float32),
            }

            roll_r_loss = 0.0
            for k_step in range(K_roll):
                current_indices = i0s + H - 1 + k_step
                true_next_indices = current_indices + 1
                if np.max(true_next_indices) >= N: break
                rs = model(hists)
                batch_controls = {
                    'acc': torch.tensor([ctrl[i]['acc'] for i in current_indices], device=device, dtype=torch.float32),
                    'delta_cmd': torch.tensor([ctrl[i]['delta_cmd'] for i in current_indices], device=device, dtype=torch.float32),
                }
                base_next_states = simulate_step_torch(batch_states, batch_controls, dt)
                corrected_next_states = base_next_states.copy()
                corrected_next_states['r_z'] += rs.squeeze(-1)
                true_next_rz = torch.tensor([meas[i]['r_z'] for i in true_next_indices], device=device, dtype=torch.float32)
                roll_r_loss += torch.nn.functional.mse_loss(corrected_next_states['r_z'], true_next_rz, reduction='sum')
                new_feats = feat_tensor_torch(corrected_next_states, batch_controls)
                hists = torch.cat([hists[:, 1:, :], new_feats.unsqueeze(1)], dim=1)
                batch_states = corrected_next_states
            roll_r = roll_r_loss / (B * K_roll) if K_roll > 0 else 0.0
            loss = elbo_loss + args.rz_weight * roll_r
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model.opt.step()
            pred_unnorm = model.lik(dist_norm).mean * model.y_std + model.y_mean
            tot_elbo += elbo_loss.item()
            mse_unnorm = torch.mean(((pred_unnorm - (by * model.y_std + model.y_mean))**2)).item()
            tot_mse  += mse_unnorm
            pbar.set_postfix(elbo=f"{elbo_loss.item():.2f}", mse=f"{mse_unnorm:.4f}", roll_mse=f"{roll_r.item():.4f}")
    # Validation loop omitted for brevity

# ----------------------- MAIN -----------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train a modular residual dynamics model.")
    # --- General arguments ---
    p.add_argument("--csv",       required=True, help="Path to the training data CSV.")
    p.add_argument("--epochs",    type=int,   default=200)
    p.add_argument("--batch",     type=int,   default=32)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--hist",      type=int,   default=10, help="History window size.")
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--wd",        type=float, default=1e-4)
    p.add_argument("--save",      default="residual_model.pt", help="Base name for the saved model file.")

    # --- Model-specific arguments ---
    p.add_argument("--encoder",   type=str,   required=True, choices=["transformer", "lstm", "gru"], help="Type of sequence encoder to use.")
    p.add_argument("--dmod",      type=int,   default=32, help="Model dimension (feature size).")
    p.add_argument("--ind",       type=int,   default=150, help="Number of inducing points for SVGP.")
    
    # --- Rollout arguments ---
    p.add_argument("--k_roll",    type=int,   default=5, help="Number of steps for rollout loss.")
    p.add_argument("--rz_weight", type=float, default=1.0, help="Weight for the rollout loss term.")

    # --- Encoder-specific arguments ---
    p.add_argument("--tf_nhead",    type=int,   default=2, help="Number of heads for Transformer.")
    p.add_argument("--tf_d_hid",  type=int,   default=128, help="Hidden dimension for Transformer's FFN.")
    p.add_argument("--tf_nlayers",  type=int,   default=2, help="Number of layers for Transformer.")
    p.add_argument("--tf_dropout",  type=float, default=0.1, help="Dropout for Transformer.")
    p.add_argument("--rnn_nlayers", type=int,   default=2, help="Number of layers for LSTM/GRU.")
    p.add_argument("--rnn_dropout", type=float, default=0.1, help="Dropout for LSTM/GRU.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Selected Encoder: {args.encoder.upper()}")

    # --- Data loading and processing ---
    df = pd.read_csv(Path(args.csv)); times = df["time"].values.astype(np.float32)
    pos_x_meas = df["pos_x"].values.astype(np.float32); pos_y_meas = df["pos_y"].values.astype(np.float32)
    acc_meas  = df["acceleration"].values.astype(np.float32); speed_meas= df["speed"].values.astype(np.float32)
    yaw_meas  = df["yaw"].values.astype(np.float32); acc_cmd   = acc_meas.copy()
    steer_cmd = np.deg2rad(df["steer_deg"].values) * K_DELTA
    if len(times) <= args.hist + 1: raise ValueError("Not enough data for history window.")
    N = len(times); dt = float(np.mean(np.diff(times))); r_z_meas = np.zeros(N, dtype=np.float32)
    r_z_meas[1:] = np.diff(yaw_meas) / dt
    meas, ctrl = [], []
    for k in range(N):
        meas.append({'pos_x': pos_x_meas[k], 'pos_y': pos_y_meas[k], 'yaw': yaw_meas[k], 'speed': speed_meas[k], 'r_z': r_z_meas[k], 'a_x': acc_meas[k], 'delta_prev': steer_cmd[k-1] if k > 0 else steer_cmd[0]})
        ctrl.append({'acc': acc_cmd[k], 'delta_cmd': steer_cmd[k]})
    X, Y = build_data(meas, ctrl, args.hist, dt)
    y_mean = Y.mean(dim=0).numpy(); y_std  = Y.std(dim=0).numpy() + 1e-8
    Y_norm = (Y - torch.from_numpy(y_mean)) / torch.from_numpy(y_std)
    ds = WindowDataset(X, Y_norm); n_val = int(len(ds) * args.val_split)
    tr_ds, vl_ds = random_split(ds, [len(ds)-n_val, n_val])
    trL = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    vlL = DataLoader(vl_ds, batch_size=args.batch)

    print("\n" + "="*50); print("Values needed for validation.py:")
    print(f"--y_mean {y_mean[0]:.8f}"); print(f"--y_std  {y_std[0]:.8f}")
    print("="*50 + "\n")

    # --- Pack encoder-specific arguments into a dictionary ---
    encoder_args = {}
    if args.encoder == 'transformer':
        encoder_args = {'nhead': args.tf_nhead, 'd_hid': args.tf_d_hid, 'nlayers': args.tf_nlayers, 'dropout': args.tf_dropout}
    else: # lstm or gru
        encoder_args = {'nlayers': args.rnn_nlayers, 'dropout': args.rnn_dropout}

    # --- Instantiate the model with the chosen encoder ---
    model = ResidualModel(
        encoder_type=args.encoder,
        encoder_args=encoder_args,
        y_mean=y_mean, y_std=y_std, 
        d_model=args.dmod, n_inducing=args.ind,
        lr=args.lr, weight_decay=args.wd, device=device
    ).to(device)
    model.mll.num_data = len(tr_ds)

    train_rollout(model, trL, vlL, meas, ctrl, args.hist, dt, device, args.epochs, args)

    # --- Save the model with a descriptive name ---
    save_path = Path(args.save)
    save_filename = save_path.with_name(f"{save_path.stem}_{args.encoder}{save_path.suffix}")
    torch.save({
        'encoder_type': args.encoder,
        'encoder': model.encoder.state_dict(),
        'gp':      model.gp.state_dict(),
        'lik':     model.lik.state_dict(),
        'args':    vars(args),
        'y_mean':  y_mean,
        'y_std':   y_std,
    }, save_filename)
    print(f"Saved final model to {save_filename}")