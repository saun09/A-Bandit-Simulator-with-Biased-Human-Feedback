import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, n_items, epsilon=0.1):
        self.epsilon = epsilon
        self.counts = np.zeros(n_items)
        self.wins = np.zeros(n_items)

    def select_item_pair(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.counts), 2, replace=False)
        else:
            win_rates = self.wins / (self.counts + 1e-5)
            best = np.argsort(win_rates)[-2:]
            return best

    def update(self, winner_idx, loser_idx):
        self.counts[winner_idx] += 1
        self.wins[winner_idx] += 1
        self.counts[loser_idx] += 1
