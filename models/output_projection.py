import torch
import torch.nn as nn

class OutputProjection(nn.Module):
    def __init__(self, embed_dim=768, output_dim=1):
        """
        Projects transformer embeddings into a final prediction.
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(embed_dim, output_dim)  # Linear mapping to output

    def forward(self, x):
        """
        Input: (Sequence Length, Batch, Embed Dim)
        Output: (Batch, Output Dim)
        """
        x = self.flatten(x)
        return self.fc(x)

# Example Usage
if __name__ == "__main__":
    dummy_input = torch.randn(10, 1, 768)  # Sequence Length = 10, Batch = 1, Embedding = 768
    projector = OutputProjection()
    output = projector(dummy_input)
    print("Final Output Shape:", output.shape)
