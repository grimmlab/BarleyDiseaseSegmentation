"""
Hyperparameter optimization and training functions for barley disease segmentation.
"""

from torch.cuda.amp import autocast, GradScaler
import mlflow
import optuna
import torch
import numpy as np
from barley_disease_segmentation.config import *
from barley_disease_segmentation.initialiser import HPOInitializer, setup_experiment_from_params
from barley_disease_segmentation.common import set_initial_mlflow_params

__all__ = [
    'get_epoch_config',
    'train_with_pruning',
    'get_early_stopping_patience',
    '_train_loop_with_mlflow',
    '_train_loop_basic',
    'objective_single'
]

def get_epoch_config(encoder_name):
    """Dynamic epoch configuration based on encoder"""
    config = {
        # Slow-converging models
        "resnet101": {
            "max_epochs": 80,
            "warmup_epochs": 15,
            "min_epochs_before_pruning": 20
        },
        "convnext_tiny": {
            "max_epochs": 60,
            "warmup_epochs": 10,
            "min_epochs_before_pruning": 15
        },

        # Standard models
        "resnet34": {
            "max_epochs": 50,
            "warmup_epochs": 8,
            "min_epochs_before_pruning": 12
        },
        "resnet50": {
            "max_epochs": 60,
            "warmup_epochs": 10,
            "min_epochs_before_pruning": 15
        },

        # Efficient models
        "efficientnet_b2": {
            "max_epochs": 40,
            "warmup_epochs": 5,
            "min_epochs_before_pruning": 8
        },
        "mobilenetv3_large_100": {
            "max_epochs": 40,
            "warmup_epochs": 5,
            "min_epochs_before_pruning": 8
        }
    }
    return config.get(encoder_name, {"max_epochs": 50, "warmup_epochs": 8, "min_epochs_before_pruning": 12})


def train_with_pruning(components, trial, epochs=15, mlflow_run=None):
    """
    Training function that supports pruning for HPO with MLflow tracking
    """
    model = components['model']
    train_loader = components['train_loader']
    val_loader = components['val_loader']
    loss_fn = components['loss_fn']
    optimizer = components['optimizer']
    scheduler = components['scheduler']
    metrics_obj = components['metrics_obj']
    params = components['params']
    scaler = GradScaler()

    best_dice = 0.0
    trial_id = f"trial_{trial.number}"
    torch.cuda.empty_cache()

    # Create a new nested run
    if mlflow_run is not None:
        with mlflow.start_run(nested=True, run_name=trial_id) as trial_run:
            set_initial_mlflow_params(trial, trial_id, params, mlflow_run.info.run_id)
            metrics = _train_loop_with_mlflow(
                model, train_loader, val_loader, loss_fn, optimizer,
                scheduler, metrics_obj, scaler, trial, epochs, params, mlflow_run=trial_run
            )
    else:
        # Without MLflow
        metrics = _train_loop_basic(
            model, train_loader, val_loader, loss_fn, optimizer,
            scheduler, metrics_obj, scaler, trial, epochs
        )

    return metrics  # Return the metrics dict


def get_early_stopping_patience(encoder_name, base_patience=15):
    """Dynamic patience based on encoder architecture"""
    patience_config = {
        # Slow-converging models
        "resnet101": 25,
        "resnet152": 30,
        "convnext_tiny": 20,
        "convnext_small": 25,

        # Standard models
        "resnet34": 15,
        "resnet50": 18,

        # Efficient, fast-converging models
        "efficientnet_b2": 12,
        "efficientnet_b3": 12,
        "mobilenetv3_large_100": 12,
    }
    return patience_config.get(encoder_name, base_patience)


def _train_loop_with_mlflow(model, train_loader, val_loader, loss_fn, optimizer,
                            scheduler, metrics_obj, scaler, trial, epochs, params, mlflow_run=None):
    """Training loop with MLflow tracking"""
    best_dice = 0.0

    # Get dynamic configuration
    encoder_name = params.get('encoder_name')
    epoch_config = get_epoch_config(encoder_name)
    epochs = epoch_config["max_epochs"]
    min_epochs_before_pruning = epoch_config["min_epochs_before_pruning"]
    patience = get_early_stopping_patience(encoder_name)

    bad_epochs = 0
    device = next(model.parameters()).device

    print(f"Using patience {patience} for encoder {encoder_name}")

    final_train_loss = 0.0
    final_learning_rate = 0.0
    epochs_completed = 0

    print("Training with MLFlow...")
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        train_metrics = []

        for batch_idx, (images, masks, metadata, bg_masks) in enumerate(train_loader):
            # Clear cache periodically to avoid memory issues during HPO
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}")

            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            bg_masks = bg_masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs = model(images)
                loss = loss_fn(outputs, masks, bg_masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            # Calculate metrics
            with torch.no_grad():
                preds = torch.softmax(outputs, dim=1) if outputs.shape[1] > 1 else torch.sigmoid(outputs)
                # Calculate training metrics the same way as validation
                train_batch_metrics = metrics_obj.get_all_metrics(preds, masks, bg_masks)
                train_metrics.append(train_batch_metrics['mean_dice'])

        epoch_loss = running_loss / len(train_loader)
        # Calculate average training metrics
        avg_train_dice = np.mean(train_metrics)

        # UPDATE CSV METRICS each epoch
        final_train_loss = epoch_loss
        final_learning_rate = optimizer.param_groups[0]['lr']
        epochs_completed = epoch + 1

        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = []
        all_preds, all_targets, all_bg = [], [], []

        with torch.no_grad(), autocast():
            for images, masks, metadata, bg_masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                bg_masks = bg_masks.to(DEVICE)

                preds = model(images)
                loss = loss_fn(preds, masks, bg_masks)
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())
                all_bg.append(bg_masks.cpu())

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_bg = torch.cat(all_bg)

        val_metrics = metrics_obj.get_all_metrics(all_preds, all_targets, all_bg)
        val_dice = val_metrics['mean_dice']
        val_iou = val_metrics['mean_iou']
        val_accuracy = val_metrics['accuracy']

        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']

        # Update best_dice
        if val_dice > best_dice:
            best_dice = val_dice
            bad_epochs = 0  # Reset counter
        else:
            bad_epochs += 1

        print(f"Epoch {epoch}:")
        print(f"  Loss: {epoch_loss:.4f}")
        print(f"  Dice: {val_dice:.4f}")
        print(f"  Best Dice: {best_dice:.4f}")
        print(f"  Learning rate: {current_lr:.2e}")
        print(f"  Bad epochs: {bad_epochs}")

        # Log metrics to MLflow
        if mlflow_run is not None:
            metrics = {
                "train_loss": epoch_loss,
                "train_dice": avg_train_dice,
                "val_loss": avg_val_loss,
                "val_dice": val_dice,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "best_dice": best_dice,
                "val_accuracy": val_accuracy
            }
            mlflow.log_metrics(metrics, step=epoch)

        # Log per-class metrics for multiclass
        if len(val_metrics['dice_per_class']) > 2:
            for i, dice in enumerate(val_metrics['dice_per_class']):
                if not np.isnan(dice):
                    mlflow.log_metric(f"dice_class_{i}", dice, step=epoch)

        # EARLY STOPPING
        if bad_epochs >= patience:
            print(f"EARLY STOPPING: No improvement for {patience} consecutive epochs")
            mlflow.log_param("early_stopped_at_epoch", epoch)
            raise optuna.exceptions.TrialPruned()

        # THRESHOLD PRUNING
        if epoch >= min_epochs_before_pruning and val_dice < 0.4:
            print(f"THRESHOLD PRUNING: Dice {val_dice:.4f} < 0.4 at epoch {epoch}")
            mlflow.log_param("threshold_pruned_at_epoch", epoch)
            raise optuna.exceptions.TrialPruned()

        # OPTUNA PRUNING
        if epoch >= min_epochs_before_pruning and trial.should_prune():
            mlflow.log_param("optuna_pruned_at_epoch", epoch)
            mlflow.log_metric("final_dice", val_dice)
            print(f"OPTUNA PRUNING at epoch {epoch}, dice: {val_dice:.4f}")
            raise optuna.exceptions.TrialPruned()

        print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Dice = {val_dice:.4f}, LR = {current_lr:.2e}")

    mlflow.log_metric("final_dice", best_dice)
    mlflow.log_param("completed_epochs", epochs)

    # RETURN METRICS DICT for CSV
    return {
        'best_dice': best_dice,
        'final_train_loss': final_train_loss,
        'final_learning_rate': final_learning_rate,
        'epochs_completed': epochs_completed
    }


def _train_loop_basic(model, train_loader, val_loader, loss_fn, optimizer,
                      scheduler, metrics_obj, scaler, trial, epochs):
    """Basic training loop without MLflow (fallback) - returns metrics"""
    best_dice = 0.0
    final_train_loss = 0.0
    final_learning_rate = 0.0
    epochs_completed = 0

    print("Training without MLFlow...")
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0

        for batch_idx, (images, masks, metadata, bg_masks) in enumerate(train_loader):
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}")

            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            bg_masks = bg_masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs = model(images)
                loss = loss_fn(outputs, masks, bg_masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        final_train_loss = epoch_loss  # Keep the last epoch's loss
        final_learning_rate = optimizer.param_groups[0]['lr']
        epochs_completed = epoch + 1

        # Validation
        model.eval()
        all_preds, all_targets, all_bg = [], [], []

        with torch.no_grad(), autocast():
            for images, masks, metadata, bg_masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                bg_masks = bg_masks.to(DEVICE)

                preds = model(images)
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())
                all_bg.append(bg_masks.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_bg = torch.cat(all_bg)

        val_metrics = metrics_obj.get_all_metrics(all_preds, all_targets, all_bg)
        val_dice = val_metrics['mean_dice']

        scheduler.step(val_dice)

        best_dice = max(best_dice, val_dice)
        trial.report(val_dice, epoch)

        if trial.should_prune():
            print(f"Trial pruned at epoch {epoch}, dice: {val_dice:.4f}")
            raise optuna.exceptions.TrialPruned()

        print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Dice = {val_dice:.4f}, LR = {final_learning_rate:.2e}")

    # Return both the best dice and additional metrics
    return {
        'best_dice': best_dice,
        'final_train_loss': final_train_loss,
        'final_learning_rate': final_learning_rate,
        'epochs_completed': epochs_completed
    }


def objective_single(trial, encoder, task, mlflow_run=None, HPO_refined=None):
    """
    Objective function for a single encoder-task combination
    """
    if HPO_refined:
        params = HPOInitializer.suggest_parameters_optimized_universal(
            trial,
            fixed_encoder=encoder,  # Fix the encoder
            fixed_task=task  # Fix the task
            )
        print(f"\n=== Trial {trial.number} for {encoder}-{task} ===")
        print(
            f"LR: {params['lr']:.2e}, Batch Size: {params['batch_size']}, Weight Decay: {params['weight_decay']:.2e}, Focal alpha: {params['focal_alpha']:.2e}, Focal gamma: {params['focal_gamma']:.2e},  Dice Weight: {params['dice_weight']:.2e} ")

    else:
        # Only optimize hyperparameters (encoder and task are fixed)
        params = HPOInitializer.suggest_parameters(
            trial,
            fixed_encoder=encoder,  # Fix the encoder
            fixed_task=task  # Fix the task
        )
        print(f"\n=== Trial {trial.number} for {encoder}-{task} ===")
        print(
            f"LR: {params['lr']:.2e}, Batch Size: {params['batch_size']}, Weight Decay: {params['weight_decay']:.2e}, Focal alpha: {params['focal_alpha']:.2e}, Focal gamma: {params['focal_gamma']:.2e},  Dice Weight: {params['dice_weight']:.2e} ")

    # Setup experiment
    components = setup_experiment_from_params(params)

    # Train with pruning
    metrics = train_with_pruning(components, trial, epochs=15, mlflow_run=mlflow_run)

    # Extract the best_dice for Optuna optimization
    best_dice = metrics['best_dice']

    # Store additional metrics in trial user attributes for later CSV saving
    trial.set_user_attr("final_train_loss", metrics.get('final_train_loss', 0.0))
    trial.set_user_attr("final_learning_rate", metrics.get('final_learning_rate', 0.0))
    trial.set_user_attr("epochs_completed", metrics.get('epochs_completed', 0))

    print(f"Trial completed. Best Dice: {best_dice:.4f}")

    return best_dice