"""
SigGNN Training Pipeline — Optimized for GPU Execution.
Includes AMP (Mixed Precision), early stopping, and adversarial generation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import time
from typing import Dict, Optional, Tuple, List

# Local imports
from config import TrainConfig


class TweedieLoss(nn.Module):
    """
    Tweedie Loss for zero-inflated forecasting (like M5).
    p=1 corresponds to Poisson (count data)
    p=2 corresponds to Gamma (strictly positive continuous)
    p=1.5 is suitable for zero-inflated continuous/semi-continuous data.
    """
    def __init__(self, p: float = 1.5):
        super().__init__()
        self.p = p

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Prevent negative predictions and zero predictions for numeric stability
        pred = torch.clamp(pred, min=1e-6)
        
        # L = -y * y_hat^(1-p)/(1-p) + y_hat^(2-p)/(2-p)
        a = target * torch.pow(pred, 1 - self.p) / (1 - self.p)
        b = torch.pow(pred, 2 - self.p) / (2 - self.p)
        
        return torch.mean(-a + b)


class PinballLoss(nn.Module):
    """Pinball loss for quantile forecasting (CRPS approximation)."""
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred should be shape (N, T, num_quantiles)
        loss = 0.0
        target = target.unsqueeze(-1)
        for i, q in enumerate(self.quantiles):
            error = target - pred[..., i:i+1]
            loss += torch.max(q * error, (q - 1) * error).mean()
        return loss / len(self.quantiles)


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
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {config.loss_fn}")

        # Learning Rate Scheduler (Cosine Annealing with Warmup)
        # We'll step this per epoch
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.max_epochs, 
            eta_min=config.lr / 100
        )
        
        # Mixed Precision Scaler
        self.scaler = GradScaler(enabled=config.use_amp)
        
        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'epoch_times': []}
        
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
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_val_loss = checkpoint['val_loss']
        self.history = checkpoint.get('history', self.history)
        print(f"   ✓ Resumed from epoch {checkpoint['epoch']} with val_loss {self.best_val_loss:.4f}")

    def generate_adversarial(
        self, 
        features: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_type: torch.Tensor, 
        targets: torch.Tensor,
        **model_kwargs
    ) -> torch.Tensor:
        """
        Fast FGSM adversarial generation for robust training.
        """
        self.model.eval()
        adv_features = features.clone().detach().requires_grad_(True)
        
        with autocast(enabled=self.config.use_amp):
            preds = self.model(adv_features, edge_index, edge_type, **model_kwargs)
            loss = self.criterion(preds, targets)
        
        self.scaler.scale(loss).backward()
        
        # FGSM step
        perturbation = self.config.adversarial_epsilon * adv_features.grad.sign()
        adv_features = features.detach() + perturbation
        
        self.model.train()
        return adv_features.detach()

    def _get_mini_batches(self, num_nodes: int) -> List[torch.Tensor]:
        """
        Generate mini-batch node indices if dataset is too large.
        If batch_size >= num_nodes, returns a single batch of all nodes.
        Note: For GNNs, full-batch is better if it fits in VRAM. This is a 
        workaround for memory limits (dropping some edges implicitly).
        """
        if self.config.batch_size >= num_nodes or self.config.batch_size <= 0:
            return [torch.arange(num_nodes, device=self.device)]
            
        indices = torch.randperm(num_nodes, device=self.device)
        batches = torch.split(indices, self.config.batch_size)
        return list(batches)

    def train_epoch(
        self, 
        features: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_type: torch.Tensor, 
        targets: torch.Tensor,
        **model_kwargs
    ) -> float:
        """Run one epoch of training."""
        self.model.train()
        total_loss = 0.0
        num_nodes = features.size(0)
        
        batches = self._get_mini_batches(num_nodes)
        
        for batch_idx in batches:
            self.optimizer.zero_grad(set_to_none=True)
            
            # Subgraph extraction (simplified: just taking node features)
            # A full implementation would use torch_geometric.utils.subgraph
            b_features = features[batch_idx]
            b_targets = targets[batch_idx]
            
            # For simplicity in this pipeline, if we batch, we just pass
            # the full graph topology but only compute loss on the batch nodes.
            # This uses more memory but is exactly mathematically correct.
            
            # ── Standard Forward Pass ──
            with autocast(enabled=self.config.use_amp):
                preds = self.model(features, edge_index, edge_type, **model_kwargs)
                loss = self.criterion(preds[batch_idx], b_targets)
            
            # ── Adversarial Component (Optional) ──
            if self.config.adversarial_training and torch.rand(1).item() < self.config.adversarial_ratio:
                adv_features = self.generate_adversarial(features, edge_index, edge_type, targets, **model_kwargs)
                with autocast(enabled=self.config.use_amp):
                    adv_preds = self.model(adv_features, edge_index, edge_type, **model_kwargs)
                    adv_loss = self.criterion(adv_preds[batch_idx], b_targets)
                loss = (loss + adv_loss) / 2.0
            
            # ── Backward Pass using Scaler ──
            self.scaler.scale(loss).backward()
            
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * len(batch_idx)
            
        return total_loss / num_nodes

    @torch.no_grad()
    def evaluate(
        self, 
        features: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_type: torch.Tensor, 
        targets: torch.Tensor,
        **model_kwargs
    ) -> float:
        """Evaluate model on validation/test set."""
        self.model.eval()
        
        with autocast(enabled=self.config.use_amp):
            preds = self.model(features, edge_index, edge_type, **model_kwargs)
            loss = self.criterion(preds, targets)
            
        return loss.item()

    def train(
        self, 
        train_data: Dict[str, torch.Tensor],
        val_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Full training loop with early stopping and checkpointing.
        """
        print(f"\n🚀 Starting training on {self.device} "
              f"(AMP: {'ON' if self.config.use_amp else 'OFF'})")
        
        # Extract tensors from dicts for faster access
        tr_features = train_data['node_features']
        edge_index = train_data['edge_index']
        edge_type = train_data['edge_type']
        tr_targets = train_data['targets']
        
        kwargs = {
            'category_ids': train_data.get('category_ids'),
            'dept_ids': train_data.get('dept_ids'),
            'historical_mean': train_data.get('historical_mean')
        }
        
        if val_data is not None:
            val_features = val_data['node_features']
            val_targets = val_data['targets']
            val_edge_index = val_data.get('edge_index', edge_index)
            val_edge_type = val_data.get('edge_type', edge_type)

        for epoch in range(1, self.config.max_epochs + 1):
            t0 = time.time()
            
            # Train
            train_loss = self.train_epoch(tr_features, edge_index, edge_type, tr_targets, **kwargs)
            
            # Step scheduler
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - t0
            
            # Validate
            if val_data is not None:
                val_loss = self.evaluate(val_features, val_edge_index, val_edge_type, val_targets, **kwargs)
            else:
                val_loss = train_loss
                
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epoch_times'].append(epoch_time)
            
            # Logging
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d}/{self.config.max_epochs:03d} | "
                      f"Time: {epoch_time:.2f}s | "
                      f"LR: {lr:.2e} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f}")
                      
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
                print(f"\n⏹ Early stopping triggered after {epoch} epochs")
                break
                
        # Load best model for return
        if self.config.checkpoint_dir and os.path.exists(self.checkpoint_path):
            self.load_checkpoint(self.checkpoint_path)
