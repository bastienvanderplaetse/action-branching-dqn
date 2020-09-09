import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BranchingQNetwork(nn.Module):
    def __init__(self, observation_space, action_space, action_bins, hidden_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(observation_space, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.value_head = nn.Linear(hidden_dim, 1)
        self.adv_heads = nn.ModuleList([nn.Linear(hidden_dim, action_bins) for i in range(action_space)])

    def forward(self, x):
        out = self.model(x)
        value = self.value_head(out)
        advs = torch.stack([l(out) for l in self.adv_heads], dim=1)

        test = advs.mean(2, keepdim=True)
        q_val = value.unsqueeze(2) + advs - advs.mean(2, keepdim=True)

        return q_val

class BranchingDQN(nn.Module):
    def __init__(self, observation_space, action_space, action_bins, target_update_freq, learning_rate, gamma, hidden_dim, td_target, device):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.action_bins = action_bins
        self.gamma = gamma

        self.policy_network = BranchingQNetwork(observation_space, action_space, action_bins, hidden_dim)
        self.target_network = BranchingQNetwork(observation_space, action_space, action_bins, hidden_dim)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optim = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.policy_network.to(device)
        self.target_network.to(device)
        self.device = device

        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.td_target = td_target

    def get_action(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            out = self.policy_network(x).squeeze(0)
            action = torch.argmax(out, dim=1)

        return action.detach().cpu().numpy()# action.numpy()

    def update_policy(self, batch):
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_masks = batch

        states = torch.tensor(batch_states).float().to(self.device)
        actions = torch.tensor(batch_actions).long().reshape(states.shape[0], -1, 1).to(self.device)
        rewards = torch.tensor(batch_rewards).float().reshape(-1, 1).to(self.device)
        next_states = torch.tensor(batch_next_states).float().to(self.device)
        masks = torch.tensor(batch_masks).float().reshape(-1, 1).to(self.device)

        current_Q = self.policy_network(states).gather(2, actions).squeeze(-1)

        with torch.no_grad():
            argmax = torch.argmax(self.policy_network(next_states), dim=2)
            max_next_Q = self.target_network(next_states).gather(2, argmax.unsqueeze(2)).squeeze(-1)
            if self.td_target == "mean":
                max_next_Q = max_next_Q.mean(1, keepdim=True)
            elif self.td_target == "max":
                max_next_Q, _ = max_next_Q.max(1, keepdim=True)

        expected_Q = rewards + max_next_Q * self.gamma * masks

        loss = F.mse_loss(expected_Q, current_Q)

        self.optim.zero_grad()
        loss.backward()

        for p in self.policy_network.parameters():
            p.grad.data.clamp_(-1., 1.)
        self.optim.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_counter = 0
            self.target_network.load_state_dict(self.policy_network.state_dict())

        return loss.detach().cpu()
