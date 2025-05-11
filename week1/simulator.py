import numpy as np
from environment import Item, UserModel

class Simulator:
    def __init__(self, n_items=10, steps=1000, user_model=None, agent=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.items = [Item(latent_quality=np.random.rand()) for _ in range(n_items)]
        self.agent = agent
        self.user = user_model
        self.steps = steps
        self.regret = []

    def run(self):
        best_quality = max(item.latent_quality for item in self.items)
        for _ in range(self.steps):
            i, j = self.agent.select_item_pair()
            winner = self.user.give_feedback(self.items[i], self.items[j])
            winner_idx = i if winner == 1 else j
            loser_idx = j if winner == 1 else i

            self.agent.update(winner_idx, loser_idx)

            regret = best_quality - self.items[winner_idx].latent_quality
            self.regret.append(regret)

        return self.regret
