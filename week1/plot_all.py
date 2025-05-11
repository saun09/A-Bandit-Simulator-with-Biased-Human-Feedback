import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_csv(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        data = [[float(x) for x in row] for row in reader]
    return np.array(data)

files = [f for f in os.listdir("results") if f.endswith(".csv")]

for file in files:
    data = load_csv(f"results/{file}")
    mean_regret = np.mean(data, axis=0)
    std_regret = np.std(data, axis=0)
    timesteps = np.arange(data.shape[1])

    plt.plot(timesteps, mean_regret, label=f"{file.replace('.csv', '')}")
    plt.fill_between(timesteps, mean_regret - std_regret, mean_regret + std_regret, alpha=0.2)

plt.xlabel("Timestep")
plt.ylabel("Cumulative Regret")
plt.title("Mean ± Std Regret under Different Feedback Biases")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/combined_regret.png")
