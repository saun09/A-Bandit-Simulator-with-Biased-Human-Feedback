import matplotlib.pyplot as plt
import seaborn as sns

def plot_regrets(results):
    plt.figure(figsize=(10, 6))
    for name, regrets in results.items():
        cumulative_regret = [sum(regrets[:i+1]) for i in range(len(regrets))]
        plt.plot(cumulative_regret, label=name)

    plt.xlabel("Timesteps")
    plt.ylabel("Cumulative Regret")
    plt.title("Bandit Algorithms under Biased Feedback")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/cumulative_regret.png")
