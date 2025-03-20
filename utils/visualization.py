# utils/visualization.py

import matplotlib.pyplot as plt

def plot_time_series(data, title="Time Series Data"):
    plt.figure(figsize=(10, 5))
    plt.plot(data, label="Stock Price")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()