import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


transform = transforms.ToTensor()
mnist_dataset = datasets.MNIST(root='.data', train= True, download= True, transform=transform)
data_loader = torch.utils.data.DataLoader( dataset= mnist_dataset, batch_size=64, shuffle=True)
def main():
    
    print("Hello from mestrado!")


if __name__ == "__main__":
    main()
