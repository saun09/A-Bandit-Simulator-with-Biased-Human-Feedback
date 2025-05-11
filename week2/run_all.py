import numpy as np
import os
import csv

from simulator import Simulator
from epsilon_greedy import EpsilonGreedyBandit
from thompson_sampling import ThompsonSamplingBandit
from environment import UserModel

# Configuration
seeds      = [0, 1, 2, 3, 4]
agents     = {
    "epsilon_greedy": EpsilonGreedyBandit,
    "thompson":       ThompsonSamplingBandit
}
user_biases = ["honest", "noisy", "random", "recent_bias"]

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

for bias in user_biases:
    for agent_name, AgentClass in agents.items():
        all_regrets     = []
        all_group_ratios = []

        for seed in seeds:
            # Initialize user, agent, simulator
            user  = UserModel(bias=bias)
            agent = AgentClass(n_items=10)
            sim   = Simulator(n_items=10,
                              steps=500,
                              user_model=user,
                              agent=agent,
                              seed=seed)

            # Run and collect
            regrets = sim.run()
            all_regrets.append(regrets)

            grp_ratio = sim.final_group_ratio()
            all_group_ratios.append(grp_ratio)

        # Save regret time-series CSV
        regret_path = f"results/{agent_name}_{bias}_regret.csv"
        with open(regret_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(all_regrets)

        # Save final group-A ratios CSV
        group_path = f"results/{agent_name}_{bias}_groups.csv"
        with open(group_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(all_group_ratios)

print("All experiments complete. CSVs in results/ folder.")

