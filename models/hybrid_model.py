import torch
import torch.nn as nn
import numpy as np
from models.quantum_layer import get_quantum_layer


class HybridQCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.quantum_layer = get_quantum_layer()  # processes 2x2 patches: 4 features

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 4 * 4, 64)   # 4 features per patch
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, n_classes)

        # Extract non-overlapping 2x2 patches from image.
    def extract_patches(self, x, patch_size=2):
        B, C, H, W = x.shape
        patches = x.unfold(2, patch_size, patch_size)\
                   .unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, -1, patch_size * patch_size)
        return patches

    def forward(self, x):
        B = x.shape[0]
        patches = self.extract_patches(x)  # (B, num_patches, 4), values in [-1, 1]

        # Rescale pixel values from [-1, 1] to [0, pi] for AngleEmbedding (rotation angles span a full half-turn on the Bloch sphere)
        scaled_patches = (patches + 1) * (np.pi / 2)

        num_patches = scaled_patches.shape[1]
        flat = scaled_patches.reshape(B * num_patches, 4) 

        # Single batched call to the quantum layer instead of one call per patch per sample to avoids B * num_patches separate circuit executions
        q_out = self.quantum_layer(flat) 
        q_out = q_out.view(B, num_patches, 4)

        side = int(num_patches ** 0.5)
        assert side * side == num_patches, \
            f"num_patches {num_patches} is not a perfect square"
        q_out = q_out.view(B, side, side, 4).permute(0, 3, 1, 2)

        x = self.pool(q_out)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
