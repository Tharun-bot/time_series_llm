# models/embedder.py

import torch
from transformers import AutoModel, AutoTokenizer

class Phi3Embedder:
    def __init__(self, model_name="microsoft/phi-2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_word_embeddings(self, words):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(words, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, embedding_dim)
        return cls_embeddings  # (num_words, embed_dim)

    
    