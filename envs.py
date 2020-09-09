import gym
import numpy as np
import torch

class BranchingEnv(gym.Wrapper):
    def __init__(self, env_name, action_bins):
        super().__init__(gym.make(env_name))
        self.name = env_name
        self.discretized = np.linspace(-1., 1., action_bins)

    def _state_to_tensor(self, state):
        return torch.tensor(state).reshape(1, -1).float()

    def reset(self):
        state = super().reset()
        return self._state_to_tensor(state)

    def step(self, actions):
        action = np.array([self.discretized[action] for action in actions])
        next_state, reward, done, infos = super().step(action)

        return self._state_to_tensor(next_state), reward, done, infos

    def set_seed(self, seed):
        super().seed(seed)
