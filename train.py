import torch
import torch.nn as nn
from tqdm import tqdm
from models.hybrid_model import HybridQCNN
from utils.dataset import get_train_loader
import os


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in tqdm(loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def main():
    USE_EDGES = False
    SUBSET_SIZE = 3000
    N_EPOCHS = 10
    BATCH_SIZE = 32
    LR = 0.01
    CLASSES = [0, 1, 2, 3]   # only 4 digits  

    train_loader = get_train_loader(batch_size=BATCH_SIZE, use_edges=USE_EDGES, subset_size=SUBSET_SIZE, classes=CLASSES)
    model = HybridQCNN(n_classes=len(CLASSES))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("checkpoints", exist_ok=True)
    for epoch in range(N_EPOCHS):
        loss, acc = train_epoch(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}: Loss={loss:.4f} | Acc={acc:.4f}")
        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pth")


if __name__ == "__main__":
    main()
