from .random_agent import RandomAgent
import numpy as np


class ProportionalFairAgent(RandomAgent):
    def __init__(self, action_space, n_ues, buffer_max_size):
        RandomAgent.__init__(self, action_space)
        self.t = 0      # Current time step

        self.K = n_ues              # Number of UEs
        self.L = buffer_max_size    # Maximum number of packets per UE buffer
        self.n = np.zeros(n_ues)    # Number of past PRB assignments for each UE

    def act(self, state, reward, done):
        s = np.reshape(state[self.K:self.K*(1 + self.L)], (self.K, self.L))  # Sizes in bits of packets in UEs' buffers
        buffer_size_per_ue = np.sum(s, axis=1)

        e = np.reshape(state[self.K*(1 + self.L):self.K*(1 + 2*self.L)], (self.K, self.L))  # Packet ages in TTIs
        o = np.max(e, axis=1)  # Age of oldest packet for each UE

        qi_ohe = np.reshape(state[self.K + 2 * self.K * self.L:5 * self.K + 2 * self.K * self.L], (self.K, 4))
        qi = np.array([np.where(r == 1)[0][0] for r in qi_ohe])  # Decode One-Hot-Encoded QIs

        # Extract packet delay budget for all UEs
        b = np.zeros(qi.shape)
        b[qi == 3] = 100
        b[qi == 2] = 150
        b[qi == 1] = 30
        b[qi == 0] = 300

        priorities = (1+o)/b * buffer_size_per_ue / (1 + self.n)

        action = np.argmax(priorities)
        self.n[action] += 1

        self.t += 1
        return action
