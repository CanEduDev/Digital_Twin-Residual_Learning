#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/trainer.py

This module contains the core logic for training the residual models.
It includes the main training loop, which handles loss calculation, backpropagation,
periodic validation via full simulation, and saving the best model checkpoint.
"""
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Import our custom modules
from .simulation import evaluate_simulation_performance, simulate_step_torch, feat_tensor_torch

def _rollout_step(model, hists, i0s, H, N, K_roll, meas, ctrl, dt, device):
    """
    A helper function to perform a vectorized rollout on a batch for training loss.

    Args:
        model: The model being trained.
        hists: The batch of historical data (B, H, F).
        i0s: The starting indices of each sample in the original measurement array.
        H (int): History window size.
        N (int): Total size of the measurement array (used for boundary checks).
        K_roll (int): Number of steps to "rollout" into the future.
        meas (list): The full list of measurement dictionaries (from the training set).
        ctrl (list): The full list of control dictionaries (from the training set).
        dt (float): Time step.
        device: PyTorch device.

    Returns:
        The calculated rollout loss for the batch.
    """
    roll_r_loss = 0.0
    B = hists.size(0)
    initial_indices = i0s + H - 1
    
    # Initialize the states for the start of the rollout
    batch_states = {
        'pos_x': torch.tensor([meas[i]['pos_x'] for i in initial_indices], device=device, dtype=torch.float32),
        'pos_y': torch.tensor([meas[i]['pos_y'] for i in initial_indices], device=device, dtype=torch.float32),
        'yaw':   torch.tensor([meas[i]['yaw'] for i in initial_indices], device=device, dtype=torch.float32),
        'speed': torch.tensor([meas[i]['speed'] for i in initial_indices], device=device, dtype=torch.float32),
        'delta_prev': torch.tensor([meas[i]['delta_prev'] for i in initial_indices], device=device, dtype=torch.float32),
        'a_x': torch.tensor([meas[i]['a_x'] for i in initial_indices], device=device, dtype=torch.float32),
    }

    for k_step in range(K_roll):
        current_indices = i0s + H - 1 + k_step
        true_next_indices = current_indices + 1
        if np.max(true_next_indices) >= N: break
        
        rs = model(hists) # Get un-normalized residual prediction
        
        batch_controls = {
            'acc': torch.tensor([ctrl[i]['acc'] for i in current_indices], device=device, dtype=torch.float32),
            'delta_cmd': torch.tensor([ctrl[i]['delta_cmd'] for i in current_indices], device=device, dtype=torch.float32),
        }
        
        base_next_states = simulate_step_torch(batch_states, batch_controls, dt)
        corrected_next_states = base_next_states.copy()
        corrected_next_states['r_z'] += rs.squeeze(-1)
        
        true_next_rz = torch.tensor([meas[i+H]['r_z'] for i in (i0s + k_step)], device=device, dtype=torch.float32)
        roll_r_loss += torch.nn.functional.mse_loss(corrected_next_states['r_z'], true_next_rz, reduction='sum')
        
        # Prepare for the next iteration
        new_feats = feat_tensor_torch(corrected_next_states, batch_controls)
        hists = torch.cat([hists[:, 1:, :], new_feats.unsqueeze(1)], dim=1)
        batch_states = corrected_next_states

    return roll_r_loss / (B * K_roll) if K_roll > 0 else 0.0

def train_model(model, train_loader, val_meas, val_ctrl, train_meas, train_ctrl, dt, device, epochs, args):
    """
    The main, unified training loop for any residual model.
    """
    scheduler = CosineAnnealingWarmRestarts(model.opt, T_0=20, T_mult=2)
    best_pos_error = float('inf')
    
    # Define save path for the best model based on its configuration
    save_path = Path(args.save)
    best_model_path = save_path.parent / f"{save_path.stem}_{args.model_type}_{args.encoder}_best.pt"
    
    train_mse_history = []
    val_pos_error_history = []
    val_epochs = []

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [{args.model_type.upper()} Train]")
        total_train_mse = 0.0

        for bx, by, idx in pbar:
            bx, by, i0s = bx.to(device), by.to(device), idx.numpy()
            model.opt.zero_grad()
            
            # --- Calculate Loss (handles both SVGP and Linear models) ---
            is_svgp = hasattr(model, 'mll')
            if is_svgp:
                dist_norm = model.gp(model.encoder(bx))
                one_step_loss = -model.mll(dist_norm, by)
                pred_unnorm = model.lik(dist_norm).mean * model.y_std + model.y_mean
            else: # Linear model
                pred_norm = model.predict_normalized(bx)
                one_step_loss = model.loss_fn(pred_norm, by)
                pred_unnorm = pred_norm * model.y_std + model.y_mean

            # --- Calculate Rollout loss using data from the training set ---
            roll_r = _rollout_step(model, bx.clone(), i0s, args.hist, len(train_meas), args.k_roll, train_meas, train_ctrl, dt, device)
            
            loss = one_step_loss + args.rz_weight * roll_r
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at epoch {ep}. Skipping batch.")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model.opt.step()
            
            # --- Log training metrics for this batch ---
            target_unnorm = by * model.y_std + model.y_mean
            mse_unnorm_sum = torch.nn.functional.mse_loss(pred_unnorm, target_unnorm, reduction='sum').item()
            total_train_mse += mse_unnorm_sum
            pbar.set_postfix(loss=f"{one_step_loss.item():.2f}", mse=f"{mse_unnorm_sum/bx.size(0):.4f}", roll_mse=f"{roll_r.item():.4f}")

        avg_train_mse = total_train_mse / len(train_loader.dataset)
        train_mse_history.append(avg_train_mse)

        # --- Periodic Validation with Full Simulation ---
        if ep % args.eval_freq == 0 or ep == epochs:
            pos_error = evaluate_simulation_performance(model, val_meas, val_ctrl, args.hist, dt, device)
            print(f"\n--- Validation Sim @ Epoch {ep} | Final Positional Error: {pos_error:.3f} m ---\n")
            val_pos_error_history.append(pos_error)
            val_epochs.append(ep)
            
            if pos_error < best_pos_error:
                best_pos_error = pos_error
                print(f"🎉 New best model! Positional Error: {best_pos_error:.3f} m. Saving to {best_model_path}...")
                
                # Create the payload to save
                save_payload = {
                    'model_type': args.model_type,
                    'encoder_type': args.encoder,
                    'encoder': model.encoder.state_dict(),
                    'args': vars(args),
                    'y_mean': model.y_mean.cpu().numpy(),
                    'y_std': model.y_std.cpu().numpy()
                }
                
                # Add model-specific weights
                if is_svgp:
                    save_payload.update({'gp': model.gp.state_dict(), 'lik': model.lik.state_dict()})
                else:
                    save_payload['regressor'] = model.regressor.state_dict()
                    
                torch.save(save_payload, best_model_path)
        
        scheduler.step()
    
    return train_mse_history, val_pos_error_history, val_epochs