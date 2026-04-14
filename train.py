"""
SigGNN Training Pipeline — Optimized for GPU Execution.
Includes AMP (Mixed Precision), early stopping, WRMSSE tracking,
and NaN-safe training loop.

UPDATED: 
- Dynamic device-type detection for autocast (CUDA/CPU)
- BF16 support for A100 GPUs via amp_dtype config
- Robust NaN recovery in WRMSSE evaluation
- Epoch-level WRMSSE tracking with NaN fallback
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
import os
import time
from typing import Dict, Optional, Tuple, List

# Local imports
from config import TrainConfig
from models.siggnn import TweedieLoss, WRMSSEAlignedLoss, BlendedLoss


class SigGNNTrainer:
    """
    GPU-optimized trainer for the SigGNN model.
    Handles data batching, AMP, learning rate scheduling, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        
        # ── Determine AMP dtype ──
        # BF16 on A100 (no overflow risk), FP16 on older GPUs
        self.amp_dtype = torch.float16  # default
        if hasattr(config, 'amp_dtype'):
            if config.amp_dtype == 'bfloat16' and torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    self.amp_dtype = torch.bfloat16
                    print(f"   ✓ Using BF16 (bfloat16) for AMP — no overflow risk!")
                else:
                    print(f"   ⚠️ BF16 requested but not supported. Falling back to FP16.")
            elif config.amp_dtype == 'float16':
                self.amp_dtype = torch.float16
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Loss Function
        if config.loss_fn == 'tweedie':
            self.criterion = TweedieLoss(p=config.tweedie_p)
        elif config.loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        elif config.loss_fn == 'huber':
            self.criterion = nn.HuberLoss(delta=1.0)
        elif config.loss_fn == 'wrmsse':
            self.criterion = WRMSSEAlignedLoss()
        elif config.loss_fn == 'blended':
            self.criterion = BlendedLoss(huber_delta=1.0)
        else:
            raise ValueError(f"Unknown loss function: {config.loss_fn}")

        # Learning Rate Scheduler — WarmRestarts gives periodic LR boosts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,           # First restart at epoch 20
            T_mult=2,         # Next restarts at 40, 80, 160...
            eta_min=config.lr / 50
        )
        
        # Mixed Precision Scaler
        # Note: GradScaler is not needed for BF16 but doesn't hurt
        use_scaler = config.use_amp and self.amp_dtype == torch.float16
        self.scaler = GradScaler(enabled=use_scaler)
        
        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.nan_count = 0
        self.history = {'train_loss': [], 'val_loss': [], 'epoch_times': [], 'lr': [], 'wrmsse': []}
        
        # Checkpointing
        if config.checkpoint_dir:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            self.checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
            
            # Resume if requested
            if config.resume_from and os.path.exists(config.resume_from):
                self.load_checkpoint(config.resume_from)

    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save model, optimizer, scaler, and scheduler states."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }, path)
        print(f"   💾 Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load states from checkpoint."""
        print(f"   📂 Loading checkpoint: {path}")
        import warnings
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_val_loss = checkpoint['val_loss']
        self.history = checkpoint.get('history', self.history)
        # Ensure wrmsse key exists in old checkpoints
        if 'wrmsse' not in self.history:
            self.history['wrmsse'] = []
        print(f"   ✓ Resumed from epoch {checkpoint['epoch']} with val_loss {self.best_val_loss:.4f}")

    def train_epoch(
        self, 
        features: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_type: torch.Tensor, 
        targets: torch.Tensor,
        **model_kwargs
    ) -> float:
        """
        Run one epoch of training.
        Uses full-batch for GNN correctness (all nodes participate in message passing).
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        # ── Full-batch forward pass ──
        with autocast(device_type=self.device_type, enabled=self.config.use_amp,
                       dtype=self.amp_dtype):
            preds = self.model(features, edge_index, edge_type, **model_kwargs)
            
            # Ensure predictions and targets are in same dtype for loss
            preds = preds.float()
            targets = targets.float()
            loss = self.criterion(preds, targets)
        
        # ── NaN detection ──
        if torch.isnan(loss) or torch.isinf(loss):
            self.nan_count += 1
            print(f"   ⚠️ NaN/Inf loss detected! (count: {self.nan_count})")
            if self.nan_count >= 3:
                for pg in self.optimizer.param_groups:
                    pg['lr'] *= 0.5
                print(f"   ⚠️ Emergency LR reduction to {self.optimizer.param_groups[0]['lr']:.2e}")
                self.nan_count = 0
            return float('nan')
        
        # ── Backward pass with gradient scaling ──
        self.scaler.scale(loss).backward()
        
        if self.config.gradient_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()

    @torch.no_grad()
    def evaluate(
        self, 
        features: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_type: torch.Tensor, 
        targets: torch.Tensor,
        **model_kwargs
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Evaluate model on validation/test set.
        Returns (loss, predictions_numpy).
        """
        self.model.eval()
        
        with autocast(device_type=self.device_type, enabled=self.config.use_amp,
                       dtype=self.amp_dtype):
            preds = self.model(features, edge_index, edge_type, **model_kwargs)
            preds = preds.float()
            targets_f = targets.float()
            loss = self.criterion(preds, targets_f)
        
        loss_val = loss.item()
        if np.isnan(loss_val) or np.isinf(loss_val):
            # Fallback: compute simple MSE
            mse = torch.mean((preds - targets_f) ** 2).item()
            loss_val = mse if not (np.isnan(mse) or np.isinf(mse)) else 0.0
        
        preds_np = preds.float().cpu().numpy()
        # Sanitize predictions
        preds_np = np.nan_to_num(preds_np, nan=0.0, posinf=0.0, neginf=0.0)
        preds_np = np.clip(preds_np, 0.0, 1000.0)
        
        return loss_val, preds_np

    def train(
        self, 
        train_data: Dict[str, torch.Tensor],
        val_data: Optional[Dict[str, torch.Tensor]] = None,
        wrmsse_evaluator=None,
        extra_train_windows: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> None:
        """
        Full training loop with multi-window training, early stopping, 
        WRMSSE tracking, and checkpointing.
        """
        print(f"\n[TRAIN] Starting training on {self.device} "
              f"(AMP: {'ON' if self.config.use_amp else 'OFF'}, "
              f"dtype: {self.amp_dtype})")
        
        # Build list of all training windows
        all_windows = [train_data]
        if extra_train_windows:
            all_windows.extend(extra_train_windows)
        
        # Extract common graph structure (shared across windows)
        edge_index = train_data['edge_index']
        edge_type = train_data['edge_type']
        
        # Prepare each window's kwargs
        window_data = []
        for w in all_windows:
            wd = {
                'features': w['node_features'],
                'targets': w['targets'],
                'kwargs': {
                    'category_ids': w.get('category_ids'),
                    'dept_ids': w.get('dept_ids'),
                    'historical_mean': w.get('historical_mean'),
                    'baseline': w.get('baseline'),
                },
            }
            window_data.append(wd)
        
        # Validation data
        val_kwargs = {}
        if val_data is not None:
            val_features = val_data['node_features']
            val_targets = val_data['targets']
            val_edge_index = val_data.get('edge_index', edge_index)
            val_edge_type = val_data.get('edge_type', edge_type)
            val_kwargs = {
                'category_ids': val_data.get('category_ids'),
                'dept_ids': val_data.get('dept_ids'),
                'historical_mean': val_data.get('historical_mean'),
                'baseline': val_data.get('baseline'),
            }

        print(f"   Training nodes: {window_data[0]['features'].shape[0]}")
        print(f"   Feature dim: {window_data[0]['features'].shape[-1]}")
        print(f"   Training windows: {len(window_data)}")
        print(f"   Edges: {edge_index.shape[1]}")
        print(f"   Loss function: {self.config.loss_fn}")
        print()

        for epoch in range(1, self.config.max_epochs + 1):
            t0 = time.time()
            
            # ── Anneal blend ratio if using BlendedLoss ──
            if isinstance(self.criterion, BlendedLoss):
                anneal_end = int(self.config.max_epochs * 0.6)
                if epoch <= anneal_end:
                    ratio = 0.3 + 0.6 * (epoch / anneal_end)
                else:
                    ratio = 0.9
                self.criterion.set_blend_ratio(ratio)
            
            # ── Multi-window training ──
            epoch_losses = []
            for wi, wd in enumerate(window_data):
                loss = self.train_epoch(
                    wd['features'], edge_index, edge_type, 
                    wd['targets'], **wd['kwargs']
                )
                if not np.isnan(loss):
                    epoch_losses.append(loss)
                
                # Free VRAM between windows
                if self.device_type == 'cuda':
                    torch.cuda.empty_cache()
            
            train_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
            
            # Step scheduler once per epoch (skip on NaN)
            if not np.isnan(train_loss):
                self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - t0
            
            # Validate
            val_loss = train_loss
            wrmsse_score = None
            if val_data is not None:
                val_loss, val_preds = self.evaluate(
                    val_features, val_edge_index, val_edge_type, val_targets, 
                    **val_kwargs
                )
                if self.device_type == 'cuda':
                    torch.cuda.empty_cache()
                
                if wrmsse_evaluator is not None and val_preds is not None:
                    try:
                        val_actuals = val_targets.cpu().numpy()
                        val_actuals = np.nan_to_num(val_actuals, nan=0.0)
                        wrmsse_score = wrmsse_evaluator.compute_wrmsse(val_preds, val_actuals)
                        # Final guard
                        if np.isnan(wrmsse_score) or np.isinf(wrmsse_score):
                            wrmsse_score = None
                    except Exception as e:
                        print(f"   ⚠️ WRMSSE computation error: {e}")
                        wrmsse_score = None
                
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epoch_times'].append(epoch_time)
            self.history['lr'].append(lr)
            self.history['wrmsse'].append(wrmsse_score if wrmsse_score is not None else float('nan'))
            
            # Logging
            if epoch % 5 == 0 or epoch == 1 or epoch <= 5:
                wrmsse_str = f" | WRMSSE: {wrmsse_score:.4f}" if wrmsse_score is not None else ""
                print(f"Epoch {epoch:03d}/{self.config.max_epochs:03d} | "
                      f"Time: {epoch_time:.1f}s | "
                      f"LR: {lr:.2e} | "
                      f"Train: {train_loss:.4f} | "
                      f"Val: {val_loss:.4f}"
                      f"{wrmsse_str}")
            
            # Skip NaN epochs for checkpointing
            if np.isnan(train_loss) or np.isnan(val_loss):
                continue
                      
            # Checkpointing & Early Stopping
            if val_loss < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if self.config.checkpoint_dir:
                    self.save_checkpoint(self.checkpoint_path, epoch, val_loss)
            else:
                self.patience_counter += 1
                
            # Save periodic checkpoints
            if self.config.checkpoint_dir and epoch % self.config.save_every == 0:
                periodic_path = os.path.join(self.config.checkpoint_dir, f'model_ep{epoch}.pt')
                self.save_checkpoint(periodic_path, epoch, val_loss)
                
            if self.patience_counter >= self.config.patience:
                print(f"\n[STOP] Early stopping triggered after {epoch} epochs")
                break
                
        # Load best model for return
        if self.config.checkpoint_dir and os.path.exists(self.checkpoint_path):
            self.load_checkpoint(self.checkpoint_path)

        # Final evaluation
        if val_data is not None:
            final_loss, final_preds = self.evaluate(
                val_features, val_edge_index, val_edge_type, val_targets, 
                **val_kwargs
            )
            print(f"\n[RESULT] Best model Val Loss: {final_loss:.4f}")
            
            if wrmsse_evaluator is not None and final_preds is not None:
                try:
                    val_actuals = val_targets.cpu().numpy()
                    val_actuals = np.nan_to_num(val_actuals, nan=0.0)
                    final_wrmsse = wrmsse_evaluator.compute_wrmsse(final_preds, val_actuals)
                    print(f"[RESULT] Final WRMSSE: {final_wrmsse:.4f}")
                    
                    hier_scores = wrmsse_evaluator.compute_hierarchical_wrmsse(final_preds, val_actuals)
                    print("\n[RESULT] Hierarchical WRMSSE Breakdown:")
                    for level, score in hier_scores.items():
                        print(f"   {level:20s}: {score:.4f}")
                except Exception as e:
                    print(f"[RESULT] WRMSSE computation failed: {e}")
                    print("[RESULT] Final Val Loss (MSE proxy): {final_loss:.4f}")
