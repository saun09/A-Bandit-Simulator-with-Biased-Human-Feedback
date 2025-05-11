from simulator import Simulator
from epsilon_greedy import EpsilonGreedyBandit
from thompson_sampling import ThompsonSamplingBandit
from environment import UserModel
from plot_results import plot_regrets

results = {}

for agent_name, agent_class in {
    "Epsilon-Greedy": EpsilonGreedyBandit,
    "Thompson Sampling": ThompsonSamplingBandit
}.items():
    user = UserModel(bias="noisy", noise=0.2)  # Change to "honest", "random", etc.
    agent = agent_class(n_items=10)
    sim = Simulator(n_items=10, steps=500, user_model=user, agent=agent)
    regrets = sim.run()
    results[agent_name] = regrets

plot_regrets(results)




