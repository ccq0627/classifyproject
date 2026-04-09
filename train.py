import os
import os.path as osp
import torch

from argparse import ArgumentParser, Namespace
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch import nn

from modular.utils import plot_result, set_seed
from modular.dataset import create_dataloader
from modular.model import TinyVGG
from modular.engine import Trainer

MANUAL_SEED = 42

def main(args: Namespace) -> None:

    train_dir = osp.join(args.data_dir, "train")
    test_dir = osp.join(args.data_dir, "test")

    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    train_transform = weights.transforms()

    transform = {
        "train": train_transform,
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    train_loader, test_loader, classes = create_dataloader(
        train_dir=train_dir, 
        test_dir=test_dir, 
        transform=transform, 
        batch_size=args.batch_size
    )
    classes_num = len(classes)
    
    # model = TinyVGG(classes_num=classes_num)

    model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=classes_num)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.heads.head.parameters():
        param.requires_grad = True

    device = args.device

    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    trainer = Trainer(
        model=model, 
        dataloader=(train_loader, test_loader), 
        loss_fn=loss_fn, 
        optimizer=optimizer, 
        device=device, 
        epochs=args.epochs
    )
    result = trainer()
    if args.plot:
        plot_result(result)


if __name__ == "__main__":

    parser = ArgumentParser(description="Train a network for image classification")
    parser.add_argument("--data_dir", type=str, default="classifyproject/data", help="Path to the data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (e.g., 'cpu' or 'cuda')")
    parser.add_argument("--plot", action="store_true", help="Plot the exp results")
    args = parser.parse_args()

    set_seed(MANUAL_SEED)

    main(args)