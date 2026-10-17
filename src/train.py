"""
Train the EcoSort classifier.

Run from the project root:
    python -m src.train --config configs/baseline.json
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data import get_loaders
from src.model import build_model

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    """Fix the random number generators so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, loss_fn, optimizer):
    """One full pass over the training data. Returns (average loss, accuracy)."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()            # clear gradients from the last batch
        logits = model(X)                # forward pass: raw scores
        loss = loss_fn(logits, y)        # how wrong were we?
        loss.backward()                  # compute gradients
        optimizer.step()                 # update the weights

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(y)

    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    """Measure the model on data without changing any weights."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss = loss_fn(logits, y)

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(y)

    return total_loss / seen, correct / seen


def train(model, train_loader, val_loader, loss_fn, optimizer, cfg, ckpt_path):
    """Run epochs until validation accuracy stops improving. Returns the history."""
    best_acc, best_epoch, bad_epochs = 0.0, 0, 0
    history = []

    for epoch in range(1, cfg["max_epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        })
        print(f"epoch {epoch:>3}  train_loss {train_loss:.4f}  train_acc {train_acc:.4f}  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")

        if val_acc > best_acc:
            # New best: save these weights, and reset the patience counter.
            best_acc, best_epoch, bad_epochs = val_acc, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"Early stop at epoch {epoch}: no improvement for {cfg['patience']} epochs.")
                break

    return history, best_acc, best_epoch


def main():
    parser = argparse.ArgumentParser(description="Train the EcoSort model.")
    parser.add_argument("--config", required=True, help="path to a JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{cfg['run_id']}_best.pt"
    log_path = LOG_DIR / f"{cfg['run_id']}.json"

    print(f"Run: {cfg['run_id']}   Device: {DEVICE}")
    set_seed(cfg["seed"])                 # set the seed before anything random happens

    # Everything below is created fresh for this run, so nothing carries over.
    train_loader, val_loader = get_loaders(batch_size=cfg["batch_size"])
    model = build_model().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    start = time.time()
    history, best_acc, best_epoch = train(
        model, train_loader, val_loader, loss_fn, optimizer, cfg, ckpt_path
    )
    seconds = round(time.time() - start, 1)

    # Save everything about this run in one file, so the log explains itself.
    result = {
        "config": cfg,
        "parameters": n_params,
        "best_val_acc": round(best_acc, 6),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "seconds": seconds,
        "checkpoint": ckpt_path.relative_to(ROOT).as_posix(),
        "history": history,
    }
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nBest validation accuracy: {best_acc:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()