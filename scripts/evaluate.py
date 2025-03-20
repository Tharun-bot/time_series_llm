# scripts/evaluate.py

import torch
from models.transformer import TimeSeriesToLanguage

def evaluate(model, test_data):
    with torch.no_grad():
        predictions = model(test_data)
    print("Evaluation Complete. Output Shape:", predictions.shape)

if __name__ == "__main__":
    # Load trained model
    model = TimeSeriesToLanguage(input_dim=30, embed_dim=2560, num_heads=4)
    
    # Dummy test data
    test_data = torch.rand((10, 30))
    evaluate(model, test_data)
