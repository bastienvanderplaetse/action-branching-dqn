import numpy as np

from torch.utils import tensorboard

from utils import save_checkpoint, save_best

class Trainer:
    def __init__(self, model, env, memory, max_steps, max_episodes, epsilon_start, epsilon_final, epsilon_decay, start_learning, batch_size, save_update_freq, output_dir):
        self.model = model
        self.env = env
        self.memory = memory
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.start_learning = start_learning
        self.batch_size = batch_size
        self.save_update_freq = save_update_freq
        self.output_dir = output_dir

    def _exploration(self, step):
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * step / self.epsilon_decay)

    def loop(self):
        state = self.env.reset()
        episode_reward = 0
        best_episode_reward = None
        all_rewards = []
        w = tensorboard.SummaryWriter()

        for step in range(self.max_steps):
            epsilon = self._exploration(step)

            if np.random.random() > epsilon:
                action = self.model.get_action(state)
            else:
                action = np.random.randint(0, self.model.action_bins, size=self.model.action_space)

            next_state, reward, done, infos = self.env.step(action)
            episode_reward += reward

            if done:
                next_state = self.env.reset()
                all_rewards.append(episode_reward)
                print("Reward on Episode {}: {}".format(len(all_rewards), episode_reward))
                w.add_scalar("reward/episode_reward", episode_reward, global_step=len(all_rewards))
                if best_episode_reward == None or episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    save_best(self.model, all_rewards, self.env.name, self.output_dir)
                episode_reward = 0

            self.memory.push((
                state.reshape(-1).numpy().tolist(),
                action,
                reward,
                next_state.reshape(-1).numpy().tolist(),
                0. if done else 1.
            ))
            state = next_state

            if step > self.start_learning:
                loss = self.model.update_policy(self.memory.sample(self.batch_size))
                w.add_scalar("loss/loss", loss, global_step=step)

            if step % self.save_update_freq == 0:
                save_checkpoint(self.model, all_rewards, self.env.name, self.output_dir)

            if len(all_rewards) == self.max_episodes:
                save_checkpoint(self.model, all_rewards, self.env.name, self.output_dir)
                break

        w.close()
