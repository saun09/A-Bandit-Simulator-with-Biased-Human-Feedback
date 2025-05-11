import numpy as np

class ThompsonSamplingBandit:
    def __init__(self, n_items):
        self.alpha = np.ones(n_items)
        self.beta = np.ones(n_items)

    def select_item_pair(self):
        samples = np.random.beta(self.alpha, self.beta)
        best = np.argsort(samples)[-2:]
        return best

    def update(self, winner_idx, loser_idx):
        self.alpha[winner_idx] += 1
        self.beta[loser_idx] += 1
