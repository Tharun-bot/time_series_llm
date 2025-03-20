# utils/data_loader.py

import yfinance as yf
import numpy as np

class TimeSeriesProcessor:
    def __init__(self, ticker, start, end, column="Close"):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.column = column
        self.data = self._load_data()

    def _load_data(self):
        df = yf.download(self.ticker, start=self.start, end=self.end)
        return df[[self.column]].dropna().values

    def instance_normalize(self):
        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)
        self.data = (self.data - mean) / (std + 1e-8)
        return self

    def create_patches(self, window_size=30, stride=1):
        patches = [
            self.data[i : i + window_size] 
            for i in range(0, len(self.data) - window_size + 1, stride)
        ]
        return np.array(patches)
