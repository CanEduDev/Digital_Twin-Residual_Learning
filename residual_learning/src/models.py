#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/models.py

This module contains all PyTorch model definitions, including the sequence
encoders (Transformer, LSTM, GRU) and the top-level residual models
(probabilistic SVGP and deterministic Linear).
"""
import torch
import gpytorch

# ----------------------- ENCODER DEFINITIONS -----------------------

class TransformerEncoder(torch.nn.Module):
    """
    A Transformer-based encoder to process sequences of vehicle states.
    """
    def __init__(self, input_dim=6, d_model=32, nhead=2, d_hid=128, nlayers=2, dropout=0.1):
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, d_model)
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.final_norm = torch.nn.LayerNorm(d_model)
        self.outp = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        h = self.proj(x)
        h = self.transformer_encoder(h)
        # Average features across the time dimension
        h = h.mean(dim=1)
        h = self.outp(h)
        return self.final_norm(h)

class LSTMEncoder(torch.nn.Module):
    """
    An LSTM-based encoder to process sequences of vehicle states.
    """
    def __init__(self, input_dim=6, d_model=32, nlayers=2, dropout=0.1):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=d_model,
            num_layers=nlayers,
            batch_first=True,
            dropout=dropout if nlayers > 1 else 0
        )
        self.final_norm = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        # h_t shape: (batch_size, sequence_length, hidden_size)
        h_t, _ = self.lstm(x)
        # Return the output of the last time step
        return self.final_norm(h_t[:, -1, :])

class GRUEncoder(torch.nn.Module):
    """
    A GRU-based encoder to process sequences of vehicle states.
    """
    def __init__(self, input_dim=6, d_model=32, nlayers=2, dropout=0.1):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=d_model,
            num_layers=nlayers,
            batch_first=True,
            dropout=dropout if nlayers > 1 else 0
        )
        self.final_norm = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        h_t, _ = self.gru(x)
        return self.final_norm(h_t[:, -1, :])


def build_encoder(encoder_type: str, d_model: int, encoder_args: dict) -> torch.nn.Module:
    """
    Factory function to build the chosen encoder.
    Simplifies model creation in the main script.
    """
    if encoder_type == 'transformer':
        return TransformerEncoder(input_dim=6, d_model=d_model, **encoder_args)
    if encoder_type == 'lstm':
        return LSTMEncoder(input_dim=6, d_model=d_model, **encoder_args)
    if encoder_type == 'gru':
        return GRUEncoder(input_dim=6, d_model=d_model, **encoder_args)
    raise ValueError(f"Unknown encoder type: {encoder_type}")


# ----------------------- SVGP LAYER -----------------------

class SVGPLayer(gpytorch.models.ApproximateGP):
    """
    The Stochastic Variational Gaussian Process layer. This component makes
    the model probabilistic, allowing it to predict uncertainty.
    """
    def __init__(self, inducing_points):
        num_tasks = 1
        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0), batch_shape=torch.Size([num_tasks])
        )
        variational_strat = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_dist, learn_inducing_locations=True
        )
        mts = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            variational_strat, num_tasks=num_tasks
        )
        super().__init__(mts)
        bs = torch.Size([num_tasks])
        self.mean_module  = gpytorch.means.ConstantMean(batch_shape=bs)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=bs) + gpytorch.kernels.RBFKernel(batch_shape=bs),
            batch_shape=bs
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ----------------------- TOP-LEVEL MODEL DEFINITIONS -----------------------

class ResidualModel_SVGP(torch.nn.Module):
    """
    The complete probabilistic residual model, combining an encoder with an SVGP head.
    """
    def __init__(self, encoder_type, encoder_args, y_mean, y_std, d_model, n_inducing, lr, weight_decay, device):
        super().__init__()
        self.device = device
        self.y_mean = torch.tensor(y_mean, dtype=torch.float32, device=device)
        self.y_std  = torch.tensor(y_std,  dtype=torch.float32, device=device)
        
        self.encoder = build_encoder(encoder_type, d_model, encoder_args).to(device)
        
        # Initialize inducing points for the GP layer
        Z = torch.randn(n_inducing, d_model, device=device)
        self.gp  = SVGPLayer(Z).to(device)
        
        self.lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1).to(device)
        self.mll = gpytorch.mlls.VariationalELBO(self.lik, self.gp, num_data=1) # num_data is set later
        
        self.opt = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            {'params': self.gp.parameters(),      'lr': lr, 'weight_decay': weight_decay},
            {'params': self.lik.parameters(),     'lr': lr}
        ])

    def forward(self, x):
        """
        Defines the forward pass for inference. Returns the un-normalized residual prediction.
        """
        z = self.encoder(x)
        dist = self.gp(z)
        pred_norm = self.lik(dist).mean
        return pred_norm * self.y_std + self.y_mean

class ResidualModel_Linear(torch.nn.Module):
    """
    The complete deterministic residual model, combining an encoder with a simple Linear head.
    Used for ablation studies to prove the value of the probabilistic approach.
    """
    def __init__(self, encoder_type, encoder_args, y_mean, y_std, d_model, lr, weight_decay, device):
        super().__init__()
        self.device = device
        self.y_mean = torch.tensor(y_mean, dtype=torch.float32, device=device)
        self.y_std  = torch.tensor(y_std,  dtype=torch.float32, device=device)
        
        self.encoder = build_encoder(encoder_type, d_model, encoder_args).to(device)
        
        self.regressor = torch.nn.Linear(d_model, 1).to(device)
        self.loss_fn = torch.nn.MSELoss()
        
        self.opt = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            {'params': self.regressor.parameters(), 'lr': lr, 'weight_decay': weight_decay}
        ])

    def forward(self, x):
        """
        Defines the forward pass for inference. Returns the un-normalized residual prediction.
        """
        z = self.encoder(x)
        pred_norm = self.regressor(z)
        return pred_norm * self.y_std + self.y_mean

    def predict_normalized(self, x):
        """
        Helper function to get the normalized prediction, used for calculating loss.
        """
        z = self.encoder(x)
        return self.regressor(z)