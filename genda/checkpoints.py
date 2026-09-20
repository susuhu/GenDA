"""Strict loading of the original GenDA checkpoint format."""
from pathlib import Path
import os
import torch


def save_checkpoint(model, optimizer, scheduler, global_step, best_val_loss, filepath):
    raw = model.module if hasattr(model, "module") else model
    checkpoint = dict(model_state_dict=raw.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                      scheduler_state_dict=scheduler.state_dict(), global_step=global_step,
                      best_val_loss=best_val_loss)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint.get("global_step", 0), checkpoint.get("best_val_loss", float("inf"))
