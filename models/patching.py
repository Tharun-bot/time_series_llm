# models/patching.py

import torch.nn as nn

class PatchEmbedder(nn.Module):
    def __init__(self, window_size, embed_dim):
        super().__init__()
        self.projection = nn.Linear(window_size, embed_dim)

    def forward(self, x):
        x = x.squeeze(-1)  # Remove last dimension -> (num_patches, window_size)
        return self.projection(x)  # Output shape: (num_patches, embed_dim)

    
