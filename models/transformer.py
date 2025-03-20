# models/transformer.py

import torch.nn as nn
import torch
from models.patching import PatchEmbedder

class TimeSeriesToLanguage(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads=4):
        super().__init__()
        self.patch_embedder = PatchEmbedder(input_dim, embed_dim)
        self.text_linear = nn.Linear(embed_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, time_series_patches, text_embeddings):
        time_series_embeddings = self.patch_embedder(time_series_patches)
        time_series_embeddings = time_series_embeddings.unsqueeze(1)  

        text_prototypes = self.text_linear(text_embeddings).unsqueeze(1)

        # Ensure shape compatibility for MultiheadAttention
        time_series_embeddings = time_series_embeddings.permute(1, 0, 2)
        text_prototypes = text_prototypes.permute(1, 0, 2)

        # Ensure sequence lengths match
        min_seq_len = min(time_series_embeddings.shape[1], text_prototypes.shape[1])
        time_series_embeddings = time_series_embeddings[:, :min_seq_len, :]
        text_prototypes = text_prototypes[:, :min_seq_len, :]

        attn_output, _ = self.attention(time_series_embeddings, text_prototypes, text_prototypes)
        return self.output_linear(attn_output.squeeze(0))  

