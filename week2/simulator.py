import numpy as np
from environment import Item, UserModel

class Simulator:
    def __init__(self, n_items=10, steps=1000, user_model=None, agent=None, seed=None):
        # Reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Generate items with latent quality
        self.items = [Item(latent_quality=np.random.rand()) for _ in range(n_items)]
        self.agent = agent
        self.user = user_model
        self.steps = steps

        # For regret tracking
        self.regret = []

        # For fairness tracking: randomly assign each item to group 'A' or 'B'
        self.groups = np.random.choice(['A', 'B'], size=n_items)
        self.selection_count = {'A': 0, 'B': 0}

    def run(self):
        # Best possible quality (oracle) for regret computation
        best_quality = max(item.latent_quality for item in self.items)

        for _ in range(self.steps):
            # 1. Agent selects a pair of item indices
            i, j = self.agent.select_item_pair()

            # 2. User gives preference feedback (1 means first wins)
            winner_flag = self.user.give_feedback(self.items[i], self.items[j])
            winner_idx = i if winner_flag == 1 else j
            loser_idx  = j if winner_flag == 1 else i

            # 3. Update the bandit algorithm
            self.agent.update(winner_idx, loser_idx)

            # 4. Track group selection counts
            grp = self.groups[winner_idx]
            self.selection_count[grp] += 1

            # 5. Compute and store instantaneous regret
            regret = best_quality - self.items[winner_idx].latent_quality
            self.regret.append(regret)

        return self.regret

    def final_group_ratio(self):
        """Return fraction of selections that went to group 'A'."""
        total = self.selection_count['A'] + self.selection_count['B']
        return self.selection_count['A'] / total if total > 0 else 0.0

