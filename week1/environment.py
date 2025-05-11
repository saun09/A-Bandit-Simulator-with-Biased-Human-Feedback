import numpy as np

class Item:
    def __init__(self, latent_quality):
        self.latent_quality = latent_quality

class UserModel:
    def __init__(self, bias="honest", noise=0.1):
        self.bias = bias
        self.noise = noise

    def give_feedback(self, item_a, item_b):
        # Honest preference
        if self.bias == "honest":
            prob = 1 if item_a.latent_quality > item_b.latent_quality else 0
        elif self.bias == "noisy":
            delta = item_a.latent_quality - item_b.latent_quality
            prob = 1 / (1 + np.exp(-delta / self.noise))  # Logistic
        elif self.bias == "random":
            prob = 0.5
        elif self.bias == "recent_bias":
            prob = 0.7  # Always favors the most recently shown (item_b)
        else:
            prob = 0.5

        return 1 if np.random.rand() < prob else 0
