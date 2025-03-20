# main.py

from scripts.train import train
from utils.visualization import plot_time_series
from utils.data_loader import TimeSeriesProcessor

if __name__ == "__main__":
    # Load stock market data
    processor = TimeSeriesProcessor(ticker="AAPL", start="2023-01-01", end="2024-01-01")
    processor.instance_normalize()
    data = processor.data

    # Plot the data
    plot_time_series(data, title="AAPL Stock Price")

    # Train the model
    train()
