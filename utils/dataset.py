import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from skimage.feature import canny
import numpy as np


def get_edge_map(image_tensor):

    img = image_tensor.squeeze().numpy()
    img = (img + 1) / 2  # denormalize to [0, 1]
    edges = canny(img, sigma=1.0).astype(np.float32)
    edges = edges * 2 - 1  # rescale {0, 1} to {-1, 1}
    return torch.from_numpy(edges).unsqueeze(0)  # (1, H, W)


class EdgeMapDataset(torch.utils.data.Dataset):

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return get_edge_map(image), label


def _build_transform():
    return transforms.Compose([
        transforms.Resize((16, 16)),  # small size for quantum feasibility
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def get_train_loader(batch_size=32, use_edges=False, subset_size=None, classes=None):
    transform = _build_transform()
    train_set = MNIST(root='./data', train=True, download=True, transform=transform)

    if classes is not None:
        train_set = _filter_classes(train_set, classes)
    if subset_size is not None:
        train_set = Subset(train_set, range(subset_size))
    if use_edges:
        train_set = EdgeMapDataset(train_set)

    return DataLoader(train_set, batch_size=batch_size, shuffle=True)


def get_test_loader(batch_size=32, use_edges=False, subset_size=None, classes=None):
    transform = _build_transform()

    test_set = MNIST(root='./data', train=False, download=True, transform=transform)

    if classes is not None:
        test_set = _filter_classes(test_set, classes)
    if subset_size is not None:
        test_set = Subset(test_set, range(subset_size))
    if use_edges:
        test_set = EdgeMapDataset(test_set)

    return DataLoader(test_set, batch_size=batch_size, shuffle=False)


def _filter_classes(dataset, classes):
    targets = dataset.targets
    mask = torch.isin(targets, torch.tensor(classes))
    indices = torch.where(mask)[0].tolist()
    remap = {c: i for i, c in enumerate(classes)}
    filtered = Subset(dataset, indices)

    class RemappedDataset(torch.utils.data.Dataset):
        def __init__(self, base, remap):
            self.base = base
            self.remap = remap
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            img, label = self.base[idx]
            return img, self.remap[label]

    return RemappedDataset(filtered, remap)