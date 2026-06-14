import torch
from models.hybrid_model import HybridQCNN
from utils.dataset import get_test_loader
from utils.visualization import show_predictions


def main():
    USE_EDGES = False
    CHECKPOINT = "checkpoints/epoch_10.pth"
    BATCH_SIZE = 32
    CLASSES = [0, 1, 2, 3]

    model = HybridQCNN(n_classes=len(CLASSES))
    model.load_state_dict(torch.load(CHECKPOINT))
    model.eval()

    loader = get_test_loader(batch_size=BATCH_SIZE, use_edges=USE_EDGES, classes=CLASSES)
    show_predictions(model, loader)


if __name__ == "__main__":
    main()
