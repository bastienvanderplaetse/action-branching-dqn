import importlib
import torch

import utils

from config import Configuration
from memoryreplay import MemoryReplay
from models import BranchingDQN
from trainer import Trainer

if __name__ == '__main__':
    args = utils.parse_arguments()

    # Get configuration
    config = Configuration(args.configuration)

    seed = utils.fix_seed(config.seed)

    # Prepare environment
    env_module = importlib.import_module('envs')
    env_class = getattr(env_module, config.env_class)
    env = env_class(config.env_name, config.action_bins)
    env.set_seed(seed)

    # Global initialization
    torch.cuda.init()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")


    # Information about environments
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    # Prepare Experience Memory Replay
    memory = MemoryReplay(config.capacity)

    # Prepare agent
    agent = BranchingDQN(
        observation_space=observation_space,
        action_space=action_space,
        action_bins=config.action_bins,
        target_update_freq=config.target_update_freq,
        learning_rate=config.lr,
        gamma=config.gamma,
        hidden_dim=config.hidden_dim,
        td_target=config.td_target,
        device=device
    )

    # Prepare Trainer
    trainer = Trainer(
        model=agent,
        env=env,
        memory=memory,
        max_steps=config.max_steps,
        max_episodes=config.max_episodes,
        epsilon_start=config.epsilon_start,
        epsilon_final=config.epsilon_final,
        epsilon_decay=config.epsilon_decay,
        start_learning=config.start_learning,
        batch_size=config.batch_size,
        save_update_freq=config.save_update_freq,
        output_dir=config.output_dir
    )

    # Train
    trainer.loop()
