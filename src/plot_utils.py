# src/utils/plot_utils.py
import os
import matplotlib.pyplot as plt
import json

def plot_history(history, out_dir="outputs"):
    # Eğer path string olarak geldiyse JSON'dan yükle
    if isinstance(history, str):
        with open(history, "r") as f:
            history = json.load(f)

    os.makedirs(out_dir, exist_ok=True)

    # ---- Train vs Valid Dice ----
    train_dice = [e["dice"] for e in history["train"]]
    valid_dice = [e["dice"] for e in history["valid"]]
    plt.figure(figsize=(8,6))
    plt.plot(train_dice, label="Train Dice")
    plt.plot(valid_dice, label="Valid Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.title("Train vs Valid Dice")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{out_dir}/dice_curve.png")
    plt.close()

    # ---- Train vs Valid Loss ----
    train_loss = [e["loss"] for e in history["train"]]
    valid_loss = [e["loss"] for e in history["valid"]]
    plt.figure(figsize=(8,6))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(valid_loss, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Valid Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{out_dir}/loss_curve.png")
    plt.close()

    # ---- Per-class Dice (Valid) ----
    num_classes = len(history["valid"][0]["per_class_dice"])
    plt.figure(figsize=(8,6))
    for i in range(num_classes):
        plt.plot([e["per_class_dice"][i] for e in history["valid"]], label=f"Class {i+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.title("Per-Class Dice (Valid)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{out_dir}/per_class_dice.png")
    plt.close()

    # ---- Learning Rate ----
    lr_vals = [e["lr"] for e in history["train"]]
    plt.figure(figsize=(8,6))
    plt.plot(lr_vals, label="LR")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{out_dir}/lr_curve.png")
    plt.close()

    print(f"✅ Training plots saved under {out_dir}")
