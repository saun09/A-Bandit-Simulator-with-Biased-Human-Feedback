import numpy as np
from simulator import Simulator
from epsilon_greedy import EpsilonGreedyBandit
from thompson_sampling import ThompsonSamplingBandit
from environment import UserModel
import os
import csv

seeds = [0, 1, 2, 3, 4]
agents = {
    "epsilon_greedy": EpsilonGreedyBandit,
    "thompson": ThompsonSamplingBandit
}
user_biases = ["honest", "noisy", "random", "recent_bias"]

os.makedirs("results", exist_ok=True)

for bias in user_biases:
    for agent_name, AgentClass in agents.items():
        all_regrets = []
        for seed in seeds:
            user = UserModel(bias=bias)
            agent = AgentClass(n_items=10)
            sim = Simulator(n_items=10, steps=500, user_model=user, agent=agent, seed=seed)
            regrets = sim.run()
            all_regrets.append(regrets)
                 all_regrets.append(regrets)

        # Save to CSV
        file_path = f"results/{agent_name}_{bias}.csv"
        with open(file_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(all_regrets)
