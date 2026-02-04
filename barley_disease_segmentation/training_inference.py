"""
Training module for barley disease segmentation models.
"""

from datetime import datetime
import os
import glob
import re
from pathlib import Path
from barley_disease_segmentation.config import *
from barley_disease_segmentation.loss import FocalDiceLoss
import mlflow
import torch
import numpy as np

__all__ = ['TrainingModule']

class TrainingModule:
    """Training module for handling model training and checkpoint management."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.best_hparams = {}
        self.training_history = {}

    def retrain_final_model(self, epochs=100, save_suffix="final"):
        """Final retraining on train+val data without validation during training"""
        print("Starting final retraining on combined dataset...")

        combined_dataset = self.pipeline.create_combined_dataset()

        train_loader = torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=self.pipeline.best_hparams['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        combined_dataset.mean, combined_dataset.std = combined_dataset.compute_mean_and_std(train_loader, 'imagenet')

        model = self._create_model()

        loss_fn = FocalDiceLoss(
            dice_weight=self.pipeline.best_hparams['dice_weight'],
            alpha=self.pipeline.best_hparams['focal_alpha'],
            gamma=self.pipeline.best_hparams['focal_gamma'],
            class_weights=combined_dataset.class_weights.to(self.pipeline.device)
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.pipeline.best_hparams['lr'],
            weight_decay=self.pipeline.best_hparams['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        save_dir = self._get_save_path("utils", "final_models")  # now the checkpoints should go to utils
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_name = f"final_{self.pipeline.task_name}_{self.pipeline.best_hparams['encoder_name']}_{timestamp}"
        if save_suffix:
            model_name += f"_{save_suffix}"
        save_path = save_dir / f"{model_name}.pth"

        if self.pipeline.mlflow_experiment:
            mlflow.set_experiment(self.pipeline.mlflow_experiment)
            with mlflow.start_run(run_name=model_name) as main_run:
                self.pipeline.mlflow_run = main_run
                mlflow.log_params(self.pipeline.best_hparams)
                mlflow.log_param("final_epochs", epochs)
                mlflow.log_param("training_samples", len(combined_dataset))
                mlflow.log_param("combined_genotypes", len(self.pipeline.TRAIN_GENOTYPES + self.pipeline.VAL_GENOTYPES))

                final_losses = self._train_final_model(
                    model, train_loader, loss_fn, optimizer, scheduler, epochs, save_path, mlflow_logging=True
                )
        else:
            final_losses = self._train_final_model(
                model, train_loader, loss_fn, optimizer, scheduler, epochs, save_path, mlflow_logging=False
            )

        best_model_path = self._select_best_checkpoint(save_dir)
        self.pipeline.model = best_model_path
        print(f"Final retraining completed!")
        print(f"Model saved to: {save_path}")

        save_dir = self._get_save_path("utils", "final_models")
        save_dir.mkdir(parents=True, exist_ok=True)

        return best_model_path

    def _create_model(self):
        """Create model with best HPO parameters"""
        task_mapping = {
            "multiclass": "multiclass",
            "binary_rust": "binary",
            "binary_ram": "binary"
        }
        model_task = task_mapping[self.pipeline.best_hparams['task']]

        if self.pipeline.best_hparams['task'] == "multiclass":
            num_classes = 3
        else:
            num_classes = 2

        model = self.pipeline.model_class(
            encoder_name=self.pipeline.best_hparams['encoder_name'],
            num_classes=num_classes,
            task_type=model_task,
            bottleneck_dropout_rate=self.pipeline.best_hparams['bottleneck_dropout'],
            decoder_dropout_rate=self.pipeline.best_hparams['decoder_dropout']
        ).to(self.pipeline.device)

        if torch.cuda.device_count() > 1:
            print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model, device_ids=list(range(NUM_GPUS)))

        return model

    def _get_save_path(self, base_dir_type="results", subfolder="final_models"):
        """Get the appropriate save path based on task and encoder"""
        try:
            task_paths = {
                "multiclass": MULTICLASS_PATHS,
                "binary_ram": BINARY_RAM_PATHS,
                "binary_rust": BINARY_RUST_PATHS
            }.get(self.pipeline.task_name.lower())

            if not task_paths:
                raise ValueError(f"Unknown task name: {self.pipeline.task_name}")

            encoder_mapping = {
                'resnet34': 'Resnet',
                'resnet50': 'Resnet',
                'resnet101': 'Resnet',
                'convnext_tiny': 'Convnext',
                'convnext_small': 'Convnext',
                'convnext_base': 'Convnext',
                'efficientnet_b0': 'Efficientnet',
                'efficientnet_b1': 'Efficientnet',
                'efficientnet_b2': 'Efficientnet',
                'efficientnet_b3': 'Efficientnet',
                'efficientnet_b4': 'Efficientnet'
            }

            folder_name = encoder_mapping.get(self.pipeline.best_hparams['encoder_name'])
            base_dir = task_paths[base_dir_type][folder_name]
            full_path = Path(base_dir) / subfolder
            full_path.mkdir(parents=True, exist_ok=True)

            return full_path

        except Exception as e:
            print(f"ERROR in _get_save_path: {e}")
            fallback_path = Path("debug_results") / subfolder
            fallback_path.mkdir(parents=True, exist_ok=True)
            print(f"Using fallback path: {fallback_path}")
            return fallback_path

    def _select_best_checkpoint(self, save_dir):
        """
        Smart model selection without HPO history
        Uses only the training curves from final training
        """
        epochs, losses = self._load_final_training_curves(save_dir)

        if not epochs:
            print("  No training curves found, using final epoch")
            return self._get_latest_checkpoint(save_dir)

        # Use sophisticated plateau detection alone
        best_epoch = self._find_robust_plateau(epochs, losses)

        print(f" Model Selection:")
        print(f"   Available epochs: {min(epochs)}-{max(epochs)}")
        print(f"   Selected: epoch {best_epoch}")
        print(f"   Loss at selection: {losses[epochs.index(best_epoch)]:.4f}")

        return self._get_checkpoint_at_epoch(save_dir, best_epoch)

    def _find_robust_plateau(self, epochs, losses):
        """
        More robust plateau detection that doesn't need HPO reference
        """

        if len(losses) < 8:
            # Not enough data, use 75% of available epochs
            return epochs[int(len(epochs) * 0.75)]

        # Smooth the loss curve
        window = min(3, len(losses) // 4)
        smoothed = [np.mean(losses[max(0, i - window):i + 1]) for i in range(len(losses))]

        # Calculate improvements
        improvements = []
        for i in range(1, len(smoothed)):
            improvement = smoothed[i - 1] - smoothed[i]  # positive = improvement
            improvements.append(improvement)

        # Find where improvements become consistently small
        patience = 3  # Number of consecutive small improvements to consider as plateau
        small_improvement_count = 0

        for i in range(len(improvements)):
            # Small improvement threshold: less than 1% of total improvement so far
            total_improvement = smoothed[0] - smoothed[i]
            improvement_threshold = total_improvement * 0.01

            if improvements[i] < improvement_threshold and improvements[i] < 0.001:
                small_improvement_count += 1
            else:
                small_improvement_count = 0

            if small_improvement_count >= patience:
                return epochs[i]  # Return the epoch where plateau was detected

        # No clear plateau found - use 85% of training as heuristic
        heuristic_epoch = epochs[int(len(epochs) * 0.85)]
        print(f"   No clear plateau detected, using heuristic: epoch {heuristic_epoch}")
        return heuristic_epoch

    def _load_final_training_curves(self, save_dir):
        """Load epoch and loss data from all checkpoints"""

        checkpoint_files = glob.glob(os.path.join(save_dir, "checkpoint_epoch_*.pth"))
        epochs = []
        losses = []

        for file_path in sorted(checkpoint_files):
            try:
                # Extract epoch from filename
                epoch_match = re.search(r'epoch_(\d+)', os.path.basename(file_path))
                if epoch_match:
                    epoch = int(epoch_match.group(1))

                    # Load checkpoint to get loss
                    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
                    loss = checkpoint.get('loss', float('inf'))

                    epochs.append(epoch)
                    losses.append(loss)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue

        return epochs, losses

    def _get_checkpoint_at_epoch(self, save_dir, target_epoch):
        """Find checkpoint closest to target epoch"""

        checkpoint_files = glob.glob(os.path.join(save_dir, "checkpoint_epoch_*.pth"))
        best_diff = float('inf')
        best_checkpoint = None

        for file_path in checkpoint_files:
            epoch_match = re.search(r'epoch_(\d+)', os.path.basename(file_path))
            if epoch_match:
                epoch = int(epoch_match.group(1))
                diff = abs(epoch - target_epoch)
                if diff < best_diff:
                    best_diff = diff
                    best_checkpoint = file_path

        return best_checkpoint if best_checkpoint else self._get_latest_checkpoint(save_dir)

    def _get_latest_checkpoint(self, save_dir):
        """Fallback: get most recent checkpoint"""

        checkpoint_files = glob.glob(os.path.join(save_dir, "checkpoint_epoch_*.pth"))
        if not checkpoint_files:
            raise ValueError(f"No checkpoints found in {save_dir}")

        latest = max(checkpoint_files, key=os.path.getctime)
        print(f" Using latest checkpoint: {os.path.basename(latest)}")
        return latest

    def _train_final_model(self, model, train_loader, loss_fn, optimizer, scheduler, epochs, save_path,
                           mlflow_logging=True):
        """Final training loop - no validation, just training until convergence"""
        scaler = torch.cuda.amp.GradScaler() if self.pipeline.device.type == 'cuda' else None
        train_losses = []
        best_loss = float('inf')

        print(f"Attempting to create directory: {save_path.parent}")
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"ERROR creating directory: {e}")
            save_path = Path(f"debug_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pth")
            print(f"Using fallback save path: {save_path}")

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (images, masks, metadata, bg_masks) in enumerate(train_loader):
                images = images.to(self.pipeline.device, non_blocking=True)
                masks = masks.to(self.pipeline.device, non_blocking=True)
                bg_masks = bg_masks.to(self.pipeline.device, non_blocking=True)
                if bg_masks is not None:
                    valid_pixels = (~bg_masks).sum()
                    if valid_pixels == 0:
                        return torch.tensor(0.0, device=DEVICE, requires_grad=True)

                optimizer.zero_grad(set_to_none=True)

                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = loss_fn(outputs, masks, bg_masks)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = loss_fn(outputs, masks, bg_masks)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if batch_idx % 20 == 0:
                    print(f"  Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / num_batches
            train_losses.append(avg_epoch_loss)
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()

            if mlflow_logging:
                mlflow.log_metric("train_loss", avg_epoch_loss, step=epoch)
                mlflow.log_metric("learning_rate", current_lr, step=epoch)

            print(f"Epoch {epoch:03d}/{epochs}: Train Loss = {avg_epoch_loss:.4f}, LR = {current_lr:.2e}")

            if epoch % 5 == 0:
                checkpoint_path = save_path.parent / f"checkpoint_epoch_{epoch}.pth"
                self._save_checkpoint(model, optimizer, scheduler, scaler, epoch, avg_epoch_loss, checkpoint_path)

            if epoch > 20 and avg_epoch_loss < 0.01:
                print(f"Early convergence at epoch {epoch} (loss < 0.01)")
                break

        print(f"Attempting to save final model to: {save_path}")
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'final_epoch': epoch,
                'final_loss': avg_epoch_loss,
                'hparams': self.pipeline.best_hparams,
                'train_losses': train_losses
            }, save_path)
            print(f" Model successfully saved to: {save_path}")

            if save_path.exists():
                file_size = save_path.stat().st_size
                print(f" File verification: {file_size} bytes")
            else:
                print(f" File was not created!")

        except Exception as e:
            print(f" Error saving model: {e}")
            fallback_locations = [
                Path("debug_model.pth"),
                Path.home() / "debug_model.pth",
                Path.cwd() / f"debug_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
            ]

            for fallback_path in fallback_locations:
                try:
                    torch.save(model.state_dict(), fallback_path)
                    print(f" Model saved to fallback location: {fallback_path}")
                    break
                except Exception as fallback_e:
                    print(f" Failed to save to {fallback_path}: {fallback_e}")

        return train_losses

    def _save_checkpoint(self, model, optimizer, scheduler, scaler, epoch, loss, path):
        """Save training checkpoint"""
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'epoch': epoch,
                'loss': loss,
                'hparams': self.pipeline.best_hparams
            }, path)
            print(f" Checkpoint saved: {path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")