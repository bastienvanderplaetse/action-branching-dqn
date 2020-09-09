import importlib
import torch

import utils

from config import Configuration
from models import BranchingQNetwork

if __name__ == "__main__":
    args = utils.parse_arguments()

    # Get configuration
    config = Configuration(args.configuration)

    # Prepare environment
    env_module = importlib.import_module('envs')
    env_class = getattr(env_module, config.env_class)
    env = env_class(config.env_name, config.action_bins)

    # Global initialization
    torch.cuda.init()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")


    # Information about environments
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    # Prepare agent
    agent = BranchingQNetwork(
        observation_space=observation_space,
        action_space=action_space,
        action_bins=config.action_bins,
        hidden_dim=config.hidden_dim
    )
    agent.load_state_dict(torch.load('./runs/{}/model_state_dict_best'.format(config.output_dir)))
    agent.to(device)

    # Play
    for episode in range(config.max_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        frames = []

        while not done:
            state = state.to(device)
            with torch.no_grad():
                action = agent(state).squeeze(0)
                action = torch.argmax(action, dim=1).reshape(-1)#.numpy().reshape(-1)
            action = action.detach().cpu().numpy()#.reshape(-1)
            state, reward, done, _ = env.step(action)

            frames.append(env.render(mode="rgb_array"))
            episode_reward += reward

        print("Reward for Episode {}: {:.2f}".format(episode, episode_reward))
        utils.save_frames_as_gif(frames, config.output_dir, episode, config.dpi)

    env.close()
