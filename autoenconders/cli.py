import argparse
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

from .train import train_model
from .test import test_model


def main():
    parser = argparse.ArgumentParser(description="Treinar Autoencoder com K-Fold")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    transform = transforms.ToTensor()
    dataset = FashionMNIST("./data", download=True, transform=transform)

    # Divisão treino/teste
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Treino
    model = train_model(
        dataset=train_dataset,
        k=args.k,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

    model = model.to(device)

    # Teste
    test_model(
        model=model,
        test_dataset=test_dataset,
        device=device
    )
    
