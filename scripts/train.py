# scripts/train.py

import torch
from utils.data_loader import TimeSeriesProcessor
from models.embedder import Phi3Embedder
from models.transformer import TimeSeriesToLanguage

def train():
    # Load and preprocess time series data
    processor = TimeSeriesProcessor(ticker="AAPL", start="2023-01-01", end="2024-01-01")
    processor.instance_normalize()
    patches = processor.create_patches(window_size=30, stride=1)
    
    patches_tensor = torch.tensor(patches, dtype=torch.float32).squeeze(-1)
    print("Patches Tensor Shape:", patches_tensor.shape)

    # Get Phi-3 word embeddings
    phi3 = Phi3Embedder()
    words = ["growth", "volatility", "trend", "seasonality", "market"]
    word_embeddings = phi3.get_word_embeddings(words)
    print("Word Embeddings Shape:", word_embeddings.shape)

    # Transform time series into language representation
    model = TimeSeriesToLanguage(input_dim=30, embed_dim=2560, num_heads=4)
    language_representation = model(patches_tensor, word_embeddings)

    print("Final Language Representation Shape:", language_representation.shape)

if __name__ == "__main__":
    train()