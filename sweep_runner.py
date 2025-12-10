"""
Sweep Runner with Algorithm-Specific Episode Counts
Wrapper script for wandb sweeps that adjusts num_episodes based on algorithm
"""

import wandb
import sys
import os

# Algorithm-specific episode counts
EPISODE_MAPPING = {
    'PPO': 1000,
    'TD3': 1000,
    'SAC': 300
}

def main():
    # Initialize wandb to get the sweep config
    run = wandb.init()
    
    # Get config from wandb
    config = wandb.config
    
    # Override num_episodes based on algorithm
    algorithm = config.get('algorithm', 'PPO')
    num_episodes = EPISODE_MAPPING.get(algorithm, config.get('num_episodes', 500))
    
    print(f"Running {algorithm} with {num_episodes} episodes (algorithm-specific override)")
    
    # Import and run the training directly in this process
    # This avoids the double wandb.init() issue
    sys.path.insert(0, os.path.dirname(__file__))
    
    # Set command line arguments for gym_environment
    sys.argv = [
        'gym_environment.py',
        '--algorithm', str(config.algorithm),
        '--env', str(config.env),
        '--num-episodes', str(num_episodes),
        '--learning-rate', str(getattr(config, 'learning-rate', config.get('learning_rate', 0.0003))),
        '--gamma', str(config.gamma),
        '--batch-size', str(getattr(config, 'batch-size', config.get('batch_size', 256))),
        '--num-tests', str(getattr(config, 'num-tests', config.get('num_tests', 100)))
    ]
    
    # Add algorithm-specific parameters
    if algorithm in ['TD3', 'SAC']:
        sys.argv.extend(['--buffer-size', str(getattr(config, 'buffer-size', config.get('buffer_size', 100000)))])
    
    if algorithm == 'PPO':
        sys.argv.extend([
            '--ppo-epochs', str(getattr(config, 'ppo-epochs', config.get('ppo_epochs', 10))),
            '--entropy-coef', str(getattr(config, 'entropy-coef', config.get('entropy_coef', 0.01))),
            '--entropy-decay', str(getattr(config, 'entropy-decay', config.get('entropy_decay', 0.9))),
            '--epsilon-clip', str(getattr(config, 'epsilon-clip', config.get('epsilon_clip', 0.2))),
            '--max-grad-norm', str(getattr(config, 'max-grad-norm', config.get('max_grad_norm', 0.5))),
            '--gae-lambda', str(getattr(config, 'gae-lambda', config.get('gae_lambda', 0.95)))
        ])
    
    print(f"Arguments: {' '.join(sys.argv[1:])}")
    
    # Import and run gym_environment's main function
    # This will reuse the already-initialized wandb run
    import gym_environment
    gym_environment.main()


if __name__ == '__main__':
    main()
