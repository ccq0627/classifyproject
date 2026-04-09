from typing import Dict, List

import matplotlib.pyplot as plt
import os.path as osp
import os
import torch

def exp_report(
        epoch: int,
        epochs: int,
        result: Dict[str, List[float]]
) -> None:
    train_loss = result["train_loss"][-1]
    train_acc = result["train_acc"][-1]
    test_loss = result["test_loss"][-1]
    test_acc = result["test_acc"][-1]
    print(f"Epoch: {epoch+1}/{epochs} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.2f}% | Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")
    
def plot_result(result: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(result["train_loss"], label="Train Loss")
    plt.plot(result["test_loss"], label="Test Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(result["train_acc"], label="Train Acc")
    plt.plot(result["test_acc"], label="Test Acc")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    if not osp.exists("classifyproject/results"):
        os.makedirs("classifyproject/results")
    plt.savefig("classifyproject/results/exp_result.png")
    print("Experiment result plot saved to classifyproject/results/exp_result.png")


def set_seed(seed) -> None:

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

