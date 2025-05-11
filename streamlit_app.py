import streamlit as st
from simulator import Simulator
from epsilon_greedy import EpsilonGreedyBandit
from thompson_sampling import ThompsonSamplingBandit
from environment import UserModel
import matplotlib.pyplot as plt
import numpy as np
import os, sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "week2")
)
st.title("Bandit Simulator – Human Feedback Demo")

algo = st.selectbox("Choose algorithm", ["epsilon_greedy", "thompson"])
bias = st.selectbox("Choose user model", ["honest", "noisy", "random", "recent_bias"])
steps = st.slider("Timesteps", 100, 1000, 500)

agent = EpsilonGreedyBandit(10) if algo == "epsilon_greedy" else ThompsonSamplingBandit(10)
user = UserModel(bias=bias)
sim = Simulator(10, steps, user, agent)
regret = sim.run()

cumulative = np.cumsum(regret)
fig, ax = plt.subplots()
ax.plot(cumulative)
ax.set_title(f"{algo} under {bias} feedback")
ax.set_xlabel("Timestep")
ax.set_ylabel("Cumulative Regret")
st.pyplot(fig)
