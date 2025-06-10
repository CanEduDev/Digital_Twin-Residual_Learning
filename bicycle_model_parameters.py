"""
Residual-SVGP correction for a kinematic bicycle model
=====================================================

Usage
-----
# ❶ Train the residual GP (writes model.pt)
$ python residual_svgp.py train rosbag_aligned_data.csv

# ❷ Run the corrected simulation
$ python residual_svgp.py sim   rosbag_aligned_data.csv  model.pt
"""

from __future__ import annotations
import math, argparse, pathlib
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch, gpytorch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────────────────────────────
# 0.  Base kinematic-bicycle model (unchanged physics)
# ────────────────────────────────────────────────────────────────────────────────
L, K_DELTA, TAU_DELTA, CD = 1.0, 1.38, 0.028, 3.0e-5          # [m], [–], [s], [s m⁻¹]

def wrap_to_pi(a): return np.arctan2(np.sin(a), np.cos(a))

def load_csv(path: pathlib.Path) -> Tuple[np.ndarray, ...]:
    df = pd.read_csv(path)
    steer = np.deg2rad(df["steer_deg"].values) * K_DELTA
    return (df["time"].values, df["pos_x"].values, df["pos_y"].values,
            wrap_to_pi(df["yaw"].values), df["speed"].values,
            df["acceleration"].values, steer)

def bicycle_sim(t, px0, py0, psi0, v0, acc_cmd, steer_cmd):
    dt = float(np.mean(np.diff(t)))
    N  = t.size
    sx, sy, syaw, sv = np.empty(N), np.empty(N), np.empty(N), np.empty(N)
    x, y, psi, v, delta = float(px0), float(py0), float(psi0), float(v0), 0.0
    for k in range(N):
        sx[k], sy[k], syaw[k], sv[k] = x, y, psi, v
        delta += (dt/TAU_DELTA)*(steer_cmd[k] - delta)
        x   += v * math.cos(psi) * dt
        y   += v * math.sin(psi) * dt
        psi += (v/L) * math.tan(delta) * dt
        v   += (acc_cmd[k] - CD*v*v) * dt
        psi  = wrap_to_pi(psi)
    return sx, sy, syaw, sv


# ────────────────────────────────────────────────────────────────────────────────
# 1.  Dataset: rolling *measured* window  → residual at tail
# ────────────────────────────────────────────────────────────────────────────────
class ResidualDataset(Dataset):
    """X: (hist,7)  = [a, v, sinψ, cosψ, brake, throttle, steer]
       y: (4,)      = [δv, δψ, δx, δy] at window tail"""
    def __init__(self, csv: pathlib.Path, hist: int = 100):
        (t, px, py, psi, v, a_cmd, steer_cmd) = load_csv(csv)

        # run nominal model to obtain prediction
        sx, sy, syaw, sv = bicycle_sim(t, px[0], py[0], psi[0], v[0], a_cmd, steer_cmd)

        # residual targets (ground-truth − simulation)
        dv, dpsi, dx, dy = v - sv, wrap_to_pi(psi - syaw), px - sx, py - sy

        # features built from **measured** state (v, psi)
        feats = np.stack([a_cmd, v,
                          np.sin(psi), np.cos(psi),
                          np.zeros_like(a_cmd),           # brake (unknown)
                          np.clip(a_cmd, 0, 1),           # crude throttle proxy
                          steer_cmd], axis=1)

        Xs, ys = [], []
        for i in range(hist, len(t)):
            Xs.append(feats[i-hist:i])
            ys.append([dv[i], dpsi[i], dx[i], dy[i]])
        self.X = torch.tensor(np.stack(Xs), dtype=torch.float32)
        self.y = torch.tensor(np.stack(ys), dtype=torch.float32)

    def __len__(self):  return self.X.shape[0]
    def __getitem__(self, idx):  return self.X[idx], self.y[idx]


# ────────────────────────────────────────────────────────────────────────────────
# 2.  Lightweight 1-D CNN encoder
# ────────────────────────────────────────────────────────────────────────────────
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=7, out_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, 7, stride=4, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 7, stride=4, padding=3),          nn.ReLU())
        self.out = nn.Linear(64, out_dim)

    def forward(self, x):                       # x: (B, hist, 7)
        x = self.cnn(x.transpose(1,2))          # → (B,64,L)
        return self.out(x.mean(-1))             # global avg-pool → (B,out_dim)


# ────────────────────────────────────────────────────────────────────────────────
# 3.  Multitask SVGP (shared kernel)
# ────────────────────────────────────────────────────────────────────────────────
class ResidualGP(gpytorch.models.ApproximateGP):
    def __init__(self, z, num_tasks=4):
        q = gpytorch.variational.CholeskyVariationalDistribution(z.size(0))
        base = gpytorch.variational.VariationalStrategy(
            self, z, q, learn_inducing_locations=True)
        strat = gpytorch.variational.MultitaskVariationalStrategy(base, num_tasks)
        super().__init__(strat)

        self.mean = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.cov  = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=z.size(1)),
            batch_shape=torch.Size([num_tasks]))

    def forward(self, x):
        m, k = self.mean(x), self.cov(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(m, k)
        )


class ResidualSVGP(nn.Module):
    def __init__(self, encoder: CNNEncoder, num_inducing=128):
        super().__init__()
        self.encoder = encoder
        z = torch.randn(num_inducing, encoder.out.out_features)
        self.gp = ResidualGP(z)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=4)

    # ────────────── forward ──────────────
    def forward(self, X):  return self.gp(self.encoder(X))

    # ────────────── training ─────────────
    def fit(self, loader, lr=5e-3, epochs=200, alpha_speed=10.0,
            device='cpu', ckpt='model.pt'):
        self.to(device).train()
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.gp,
                                            num_data=len(loader.dataset))
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        for ep in range(1, epochs+1):
            elbo_sum, mse_sum = 0., 0.
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                out = self(xb)
                loss_elbo = -mll(out, yb)
                loss_mse  = torch.nn.functional.mse_loss(out.mean[:,0], yb[:,0])
                (loss_elbo + alpha_speed*loss_mse).backward()
                opt.step()
                elbo_sum += loss_elbo.item()*xb.size(0)
                mse_sum  += loss_mse.item()*xb.size(0)
            if ep % 20 == 0 or ep == 1:
                N = len(loader.dataset)
                print(f"Epoch {ep:3d}  ELBO {-elbo_sum/N:.4f}  MSE_v {mse_sum/N:.6f}")
        torch.save(self.state_dict(), ckpt)
        print(f"✓ Saved {ckpt}")

    # ────────────── load ─────────────
    def load(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device).eval()


# ────────────────────────────────────────────────────────────────────────────────
# 4.  Online corrected simulator (incremental residual gain)
# ────────────────────────────────────────────────────────────────────────────────
class CorrectedSim:
    def __init__(self, model: ResidualSVGP, hist=100, device='cpu'):
        self.net, self.hist, self.dev = model, hist, device
        self.buf: List[np.ndarray] = []

    def _push(self, row):
        self.buf.append(row)
        if len(self.buf) > self.hist: self.buf.pop(0)

    def step(self, dt, x, y, psi, v, a_cmd, steer_cmd):
        # base Euler update
        delta = steer_cmd
        x_b   = x + v*math.cos(psi)*dt
        y_b   = y + v*math.sin(psi)*dt
        psi_b = wrap_to_pi(psi + (v/L)*math.tan(delta)*dt)
        v_b   = v + (a_cmd - CD*v*v) * dt

        # assemble features from **current** measured state
        feat = np.array([a_cmd, v, math.sin(psi), math.cos(psi),
                         0.0, max(a_cmd,0.0), steer_cmd], np.float32)
        self._push(feat)

        if len(self.buf) == self.hist:
            X = torch.tensor(np.stack(self.buf)[None,...]).to(self.dev)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = self.net(X)
            δv, δψ, δx, δy = pred.mean[0].cpu().numpy()
            gain = 1.0 / self.hist                       # spread over 1 s window
            v_b   += gain * δv
            psi_b  = wrap_to_pi(psi_b + gain * δψ)
            x_b   += gain * δx
            y_b   += gain * δy

        return x_b, y_b, psi_b, v_b


# ────────────────────────────────────────────────────────────────────────────────
# 5.  CLI helpers
# ────────────────────────────────────────────────────────────────────────────────
def train(csv, hist=100):
    ds = ResidualDataset(pathlib.Path(csv), hist)
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
    net = ResidualSVGP(CNNEncoder())
    net.fit(dl, device='cuda' if torch.cuda.is_available() else 'cpu')

def simulate(csv, model_pt, hist=100):
    t, px, py, psi, v, a_cmd, steer_cmd = load_csv(pathlib.Path(csv))
    dt = float(np.mean(np.diff(t)))

    net = ResidualSVGP(CNNEncoder());  net.load(model_pt)
    sim = CorrectedSim(net, hist)

    xs, ys, psis, vs = [], [], [], []
    x, y, ya, vv = px[0], py[0], psi[0], v[0]
    for k in tqdm(range(len(t)), desc="Sim"):
        x, y, ya, vv = sim.step(dt, x, y, ya, vv, a_cmd[k], steer_cmd[k])
        xs.append(x); ys.append(y); psis.append(ya); vs.append(vv)

    sx, sy, syaw, sv = bicycle_sim(t, px[0], py[0], psi[0], v[0], a_cmd, steer_cmd)

    # RMSE
    rmse_pos  = np.sqrt(np.mean((np.hypot(px-np.array(xs), py-np.array(ys)))**2))
    rmse_yaw  = np.sqrt(np.mean((wrap_to_pi(psi-np.array(psis)))**2))
    rmse_speed= np.sqrt(np.mean((v-np.array(vs))**2))
    print(f"\nBase RMSE   pos  {np.sqrt(np.mean((np.hypot(px-sx,py-sy))**2)):.3f} m"
          f"   yaw {np.sqrt(np.mean((wrap_to_pi(psi-syaw))**2)):.3f} rad"
          f"   speed {np.sqrt(np.mean((v-sv)**2)):.3f} m/s")
    print(f"Corr RMSE   pos  {rmse_pos:.3f} m   yaw {rmse_yaw:.3f} rad   speed {rmse_speed:.3f} m/s")

    # plot
    fig = plt.figure(figsize=(13,9)); fig.suptitle("Real vs Sim vs Corrected")
    ax1 = plt.subplot2grid((2,2),(0,0), colspan=2)
    ax1.plot(px,py,'k',lw=2,label="Real");   ax1.plot(sx,sy,'b--',label="Sim")
    ax1.plot(xs,ys,'r',label="Corrected");   ax1.axis('equal'); ax1.legend(); ax1.set_title("Trajectory")
    ax2 = plt.subplot2grid((2,2),(1,0))
    ax2.plot(t,v,'k',lw=2); ax2.plot(t,sv,'b--'); ax2.plot(t,vs,'r'); ax2.set_title("Speed")
    ax3 = plt.subplot2grid((2,2),(1,1))
    ax3.plot(t,psi,'k',lw=2); ax3.plot(t,syaw,'b--'); ax3.plot(t,psis,'r'); ax3.set_title("Yaw")
    plt.tight_layout(); plt.show()


# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p_tr = sub.add_parser("train"); p_tr.add_argument("csv")
    p_si = sub.add_parser("sim");   p_si.add_argument("csv"); p_si.add_argument("model")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.csv)
    else:
        simulate(args.csv, args.model)
