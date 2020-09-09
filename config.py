import json

class Configuration:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Environment
        self.env_name = config['environment']['env_name']
        self.env_class = config['environment']['env_class']
        self.action_bins = config['environment']['action_bins']

        # Exploration
        self.epsilon_start = config['exploration']['epsilon_start']
        self.epsilon_final = config['exploration']['epsilon_final']
        self.epsilon_decay = config['exploration']['epsilon_decay']

        # Training
        self.target_update_freq = config['training']['target_update_freq']
        self.start_learning = config['training']['start_learning']
        self.lr = config['training']['learning_rate']
        self.max_steps = config['training']['max_steps']
        self.max_episodes = config['training']['max_episodes']
        self.seed = config['training']['seed']

        # Memory replay
        self.capacity = config['memory_replay']['capacity']
        self.batch_size = config['memory_replay']['batch_size']

        # Output
        self.save_update_freq = config['output']['save_update_freq']
        self.output_dir = config['output']['directory']
        self.dpi = config['output']['dpi']

        # Model
        self.td_target = config['model']['temporal_difference_target']
        assert self.td_target in ("mean", "max", "individual")
        self.gamma = config['model']['gamma']
        self.hidden_dim = config['model']['hidden_dim']

        # Device
        self.device = config['device']
