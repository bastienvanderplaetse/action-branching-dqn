import os
import numpy as np
import pandas as pd
import torch
import random

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib import animation

from scipy.ndimage.filters import gaussian_filter1d

from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('configuration', type=str, help="the name of the configuration file (JSON file)")

    return parser.parse_args()

def fix_seed(seed=None):
    if seed is None:
        seed = time.time()

    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    return seed

def _save(agent, rewards, env_name, output_dir, model_type):

    path = './runs/{}/'.format(output_dir)
    try:
        os.makedirs(path)
    except:
        pass

    torch.save(agent.policy_network.state_dict(), os.path.join(path, 'model_state_dict'+model_type))

    plt.cla()
    plt.plot(rewards, c = '#bd0e3a', alpha = 0.3)
    plt.plot(gaussian_filter1d(rewards, sigma = 5), c = '#bd0e3a', label = 'Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward')
    plt.title('Branching DDQN ({}): {}'.format(agent.td_target, env_name))
    plt.savefig(os.path.join(path, 'reward.png'))

    pd.DataFrame(rewards, columns = ['Reward']).to_csv(os.path.join(path, 'rewards.csv'), index = False)

def save_checkpoint(agent, rewards, env_name, output_dir):
    _save(agent, rewards, env_name, output_dir, "_last")

def save_best(agent, rewards, env_name, output_dir):
    _save(agent, rewards, env_name, output_dir, "_best")

def save_frames_as_gif(frames, output_dir, episode, dpi):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi), dpi=dpi)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=10)
    filename = "runs/{}/episode_{}.gif".format(output_dir, episode)
    anim.save(filename, writer='imagemagick',fps=60)
