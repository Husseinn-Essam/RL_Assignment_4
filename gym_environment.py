"""
Gymnasium Environment Runner for SAC, PPO, and TD3
This script trains and tests SAC/PPO/TD3 agents on Gymnasium environments
with Weights & Biases tracking, hyperparameter tuning, and video recording.
Corrected for CarRacing-v3 action spaces.
"""

import gymnasium as gym
import numpy as np
import torch
import wandb
import os
import argparse
from datetime import datetime
from gymnasium.wrappers import RecordVideo
import cv2

from models import SACAgent, PPOAgent, TD3Agent


class CarRacingPreprocessor(gym.ObservationWrapper):
    """Preprocessing wrapper for CarRacing: grayscale, resize, and frame stack."""
    
    def __init__(self, env, img_size=84, stack_frames=4):
        super().__init__(env)
        self.img_size = img_size
        self.stack_frames = stack_frames
        self.frames = []
        
        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(stack_frames, img_size, img_size),
            dtype=np.uint8
        )
    
    def observation(self, obs):
        # CarRacing-v3 returns frames as (H, W, 3)
        # Convert to grayscale and resize
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        
        # Add to frame stack
        self.frames.append(resized)
        if len(self.frames) > self.stack_frames:
            self.frames.pop(0)
        
        # If not enough frames yet, repeat the current frame
        while len(self.frames) < self.stack_frames:
            self.frames.append(resized)
        
        # Stack frames
        return np.stack(self.frames, axis=0)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = []
        return self.observation(obs), info


class CarRacingActionWrapper(gym.ActionWrapper):
    """
    Rescale agent actions from [-1, 1] to the valid range for CarRacing-v3.
    Agent outputs: [Steering, Gas, Brake] all in [-1, 1]
    Environment expects:
        Steering: [-1, 1]
        Gas: [0, 1]
        Brake: [0, 1]
    """
    def action(self, action):
        # Steer: [-1, 1] -> [-1, 1] (No change)
        steer = action[0]
        
        # Gas: [-1, 1] -> [0, 1]
        # Map -1 to 0, and 1 to 1 (init 0 -> 0.5)
        # This helps exploration by forcing the car to move initially
        gas = (action[1] + 1) / 2.0
        
        # Brake: [-1, 1] -> [0, 1]
        # Map -1 to 0 (clip), and 1 to 1 (init 0 -> 0)
        # This prevents the car from starting with brakes on
        brake = np.clip(action[2], 0, 1)
        
        return np.array([steer, gas, brake])


class FrameSkip(gym.Wrapper):
    """Skip frames to speed up training. Repeat action for n frames and return last observation."""
    
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        # Initialize default return values
        obs = None 
        info = {}
        terminated = False
        truncated = False
        
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        return obs, total_reward, terminated, truncated, info


def generate_run_name(algorithm, env_name, learning_rate, gamma, buffer_size, batch_size):
    """
    Generate a descriptive name for the run that includes key hyperparameters.
    """
    env_short = env_name.split('-')[0]  # e.g., CartPole from CartPole-v1
    return (f"{algorithm}_{env_short}_"
            f"lr{learning_rate}_g{gamma}_"
            f"buf{buffer_size//1000}k_bs{batch_size}")


def create_environment(env_name, render_mode=None, video_folder=None, episode_trigger=None, name_prefix=None):
    """
    Create a Gymnasium environment with optional video recording and specific wrappers.
    """
    # Handle continuous environments that require special parameters
    if env_name == 'LunarLander-v3':
        env = gym.make(env_name, continuous=True, render_mode=render_mode if video_folder else None)
    elif env_name == 'CarRacing-v3':
        # CarRacing always needs rgb_array mode since it returns pixel observations
        # Use the provided render_mode or default to 'rgb_array'
        car_render_mode = render_mode if render_mode else 'rgb_array'
        env = gym.make(env_name, continuous=True, render_mode=car_render_mode)
        
        # 1. Apply frame skip to speed up training (4x faster)
        env = FrameSkip(env, skip=4)
        
        # 2. Apply Action Wrapper to fix the [-1, 1] vs [0, 1] mismatch
        env = CarRacingActionWrapper(env)
        
        # 3. Apply preprocessing for image-based observation (Grayscale + Stack)
        env = CarRacingPreprocessor(env)
    else:
        env = gym.make(env_name, render_mode=render_mode if video_folder else None)
    
    if video_folder:
        if name_prefix is None:
            name_prefix = f"{env_name}-rl-video"
        env = RecordVideo(
            env, 
            video_folder=video_folder,
            episode_trigger=episode_trigger if episode_trigger else lambda x: True,
            name_prefix=name_prefix
        )
    
    return env


def get_action_space_info(env):
    """Get action space dimension for continuous environments."""
    return env.action_space.shape[0]





def train_agent(
    agent,
    env_name,
    run_name,
    num_episodes=1000,
    max_steps=500,
    log_interval=10,
    save_dir='models',
    use_wandb=True
):
    """
    Train the agent on the specified environment.
    """
    env = create_environment(env_name)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        
        # Print progress for first episode
        if episode == 0:
            print(f"Episode {episode + 1} started, collecting experiences...")
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Take action
            # Note: For CarRacing, the wrapper handles the conversion from [-1,1] to [0,1]
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition - use 'terminated' NOT 'done' for the done flag
            # This is critical: truncated episodes (time limit) should NOT be treated as terminal
            agent.store_transition(state, action, reward, next_state, terminated)
            
            # Train SAC/TD3 every step for better sample efficiency
            # For A2C and PPO, train() will be called at end of episode
            if hasattr(agent, 'memory'):  # SAC/TD3 has replay buffer
                loss = agent.train()
                if loss is not None:
                    episode_loss.append(loss)
                    # Print when training actually starts (buffer is full enough)
                    if episode == 0 and len(episode_loss) == 1:
                        print(f"Training started at step {step + 1} (replay buffer ready)")
            
            episode_reward += reward
            state = next_state
            
            # Print step progress for first episode
            if episode == 0 and (step + 1) % 50 == 0:
                print(f"  Step {step + 1}/{max_steps}, reward so far: {episode_reward:.2f}")
            
            if done:
                break
        
        # Train agent at end of episode (for A2C and PPO)
        if not hasattr(agent, 'memory'):  # A2C and PPO don't have replay buffer
            loss = agent.train()
            if loss is not None:
                episode_loss.append(loss)
        
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)
        
        # Logging
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_loss = np.mean(episode_losses[-log_interval:])
            entropy_or_alpha = agent.get_entropy_coef()
            
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Loss: {avg_loss:.4f}, "
                  f"Entropy/Alpha: {entropy_or_alpha:.4f}")
            
            if use_wandb:
                wandb.log({
                    'train/episode': episode + 1,
                    'train/reward': episode_reward,
                    'train/avg_reward': avg_reward,
                    'train/loss': avg_loss,
                    'train/entropy_coef': entropy_or_alpha
                }, step=episode + 1)
    
    # Save final model with descriptive name
    model_path = os.path.join(save_dir, f"{run_name}.pt")
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    env.close()
    return agent


def test_agent(
    agent,
    env_name,
    run_name,
    num_tests=100,
    max_steps=500,
    render=False,
    record_video=False,
    video_folder='videos',
    use_wandb=False,
    training_episodes=0
):
    """Test the trained agent and optionally log per-episode metrics to W&B."""

    if record_video:
        os.makedirs(video_folder, exist_ok=True)
        env = create_environment(
            env_name,
            render_mode='rgb_array',
            video_folder=video_folder,
            episode_trigger=lambda x: x % 10 == 0,
            name_prefix=run_name
        )
    else:
        env = create_environment(env_name, render_mode='human' if render else None)

    test_rewards = []
    test_durations = []

    for test_episode in range(num_tests):
        state, _ = env.reset()
        episode_reward = 0
        episode_duration = 0

        for step in range(max_steps):
            action = agent.select_action(state, epsilon=0.0)
            
            # The environment wrapper will handle the action scaling
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_duration += 1
            state = next_state

            if done:
                break

        test_rewards.append(episode_reward)
        test_durations.append(episode_duration)

        if use_wandb:
            # Use global step counter that continues from training
            global_step = training_episodes + test_episode + 1
            wandb.log({
                'test/episode': test_episode + 1,
                'test/episode_reward': episode_reward,
                'test/episode_duration': episode_duration
            }, step=global_step)

        if (test_episode + 1) % 10 == 0:
            print(f"Test Episode {test_episode + 1}/{num_tests} - "
                  f"Reward: {episode_reward:.2f}, Duration: {episode_duration}")

    env.close()

    if use_wandb:
        metrics_table = wandb.Table(columns=['episode', 'reward', 'duration'])
        for episode_idx, (reward_val, duration_val) in enumerate(zip(test_rewards, test_durations), start=1):
            metrics_table.add_data(episode_idx, reward_val, duration_val)

        wandb.log({'test_episode_metrics': metrics_table})

    results = {
        'mean_reward': np.mean(test_rewards),
        'std_reward': np.std(test_rewards),
        'mean_duration': np.mean(test_durations),
        'std_duration': np.std(test_durations),
        'min_reward': np.min(test_rewards),
        'max_reward': np.max(test_rewards)
    }
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Duration: {results['mean_duration']:.2f} ± {results['std_duration']:.2f}")
    print(f"Reward Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
    print("="*50 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train SAC/PPO/TD3 on Gymnasium environments')
    
    # Environment settings
    parser.add_argument('--env', type=str, default='LunarLander-v3',
                        choices=[
                                'LunarLander-v3', 'CarRacing-v3'],
                        help='Gymnasium environment')
    parser.add_argument('--algorithm', type=str, default='SAC',
                        choices=['SAC', 'PPO', 'TD3'],
                        help='Algorithm to use')
    
    # Training settings
    parser.add_argument('--num-episodes', '--num_episodes', type=int, default=1000, dest='num_episodes',
                        help='Number of training episodes')
    parser.add_argument('--max-steps', '--max_steps', type=int, default=1000, dest='max_steps',
                        help='Maximum steps per episode')
    parser.add_argument('--num-tests', '--num_tests', type=int, default=100, dest='num_tests',
                        help='Number of test episodes')
    
    # Hyperparameters
    parser.add_argument('--learning-rate', '--learning_rate', type=float, default=0.0003, dest='learning_rate',
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--buffer-size', '--buffer_size', type=int, default=1000000, dest='buffer_size',
                        help='Replay buffer size (for SAC/TD3)')
    parser.add_argument('--batch-size', '--batch_size', type=int, default=256, dest='batch_size',
                        help='Batch size for training')
    parser.add_argument('--hidden-sizes', '--hidden_sizes', type=int, nargs='+', default=[256, 256], dest='hidden_sizes',
                        help='Hidden layer sizes')
    
    # PPO-specific parameters
    parser.add_argument('--epsilon-clip', '--epsilon_clip', type=float, default=0.2, dest='epsilon_clip',
                        help='PPO clipping parameter (epsilon for clipped surrogate objective)')
    parser.add_argument('--ppo-epochs', '--ppo_epochs', type=int, default=10, dest='ppo_epochs',
                        help='Number of PPO optimization epochs per update')
    
    # PPO GAE parameter
    parser.add_argument('--gae-lambda', '--gae_lambda', type=float, default=0.95, dest='gae_lambda',
                        help='GAE lambda parameter for PPO (0=TD, 1=MC, 0.95 recommended)')
    
    # PPO entropy regularization
    parser.add_argument('--entropy-coef', '--entropy_coef', type=float, default=0.01, dest='entropy_coef',
                        help='Initial entropy coefficient for exploration bonus (PPO)')
    parser.add_argument('--entropy-decay', '--entropy_decay', type=float, default=0.5, dest='entropy_decay',
                        help='Fraction of training episodes (0-1) over which to linearly decay entropy to zero (PPO). Default: 0.5 (first 50%%)')
    
    # PPO gradient clipping
    parser.add_argument('--max-grad-norm', '--max_grad_norm', type=float, default=0.5, dest='max_grad_norm',
                        help='Maximum gradient norm for clipping (PPO)')
    
    # PPO continuous action parameters
    parser.add_argument('--reward-scale', '--reward_scale', type=float, default=1.0, dest='reward_scale',
                        help='Reward scaling factor (SAC/PPO/TD3). Default: 1.0. Try 0.1 for large negative rewards.')
    parser.add_argument('--action-scale', '--action_scale', type=float, default=1.0, dest='action_scale',
                        help='Action scaling factor for continuous control (PPO). Default: 1.0')
    
    # SAC-specific parameters
    parser.add_argument('--tau', type=float, default=0.005,
                        help='SAC/TD3 soft update coefficient for target networks')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='SAC entropy regularization coefficient (temperature parameter)')
    
    # TD3-specific parameters
    parser.add_argument('--policy-noise', '--policy_noise', type=float, default=0.2, dest='policy_noise',
                        help='TD3 target policy smoothing noise std')
    parser.add_argument('--noise-clip', '--noise_clip', type=float, default=0.5, dest='noise_clip',
                        help='TD3 target policy smoothing noise clip range')
    parser.add_argument('--policy-delay', '--policy_delay', type=int, default=2, dest='policy_delay',
                        help='TD3 delayed policy update frequency')
    parser.add_argument('--exploration-noise', '--exploration_noise', type=float, default=0.1, dest='exploration_noise',
                        help='TD3 exploration noise std')
    
    # PER parameters
    parser.add_argument('--use-per', action='store_true', dest='use_per',
                        help='Enable Prioritized Experience Replay (PER)')
    parser.add_argument('--per-alpha', type=float, default=0.6, dest='per_alpha',
                        help='PER alpha (how much prioritization is used, 0 = uniform)')
    parser.add_argument('--per-eps', type=float, default=1e-6, dest='per_eps',
                        help='PER epsilon to avoid zero priority')
    parser.add_argument('--per-beta', type=float, default=0.4, dest='per_beta',
                        help='Initial PER beta (importance-sampling correction)')
    
    # Other settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-wandb', '--no_wandb', action='store_true', dest='no_wandb',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--test-only', '--test_only', action='store_true', dest='test_only',
                        help='Only test, do not train')
    parser.add_argument('--load-model', '--load_model', type=str, default=None, dest='load_model',
                        help='Path to model to load')
    parser.add_argument('--record-video', '--record_video', action='store_true', dest='record_video',
                        help='Record videos during testing')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during testing')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create environment to get dimensions (initially, before any sweep overrides)
    if args.env == 'LunarLander-v3':
        temp_env = gym.make(args.env, continuous=True)
    elif args.env == 'CarRacing-v3':
        # Apply wrappers here to get correct dimensions
        temp_env = gym.make(args.env, continuous=True)
        temp_env = FrameSkip(temp_env, skip=4)
        temp_env = CarRacingActionWrapper(temp_env)
        temp_env = CarRacingPreprocessor(temp_env)
    else:
        temp_env = gym.make(args.env)
    
    # Detect if environment uses images (CNN needed)
    use_cnn = len(temp_env.observation_space.shape) > 1
    if use_cnn:
        # For image-based observations
        input_channels = temp_env.observation_space.shape[0]
        state_size = temp_env.observation_space.shape[1]  # Image size (e.g., 84)
    else:
        # For vector-based observations
        state_size = temp_env.observation_space.shape[0]
        input_channels = 1
    
    action_size = get_action_space_info(temp_env)
    temp_env.close()
    
    # Initialize W&B before constructing the agent so sweep configs can override args
    use_wandb = not args.no_wandb
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project='rl-policy-gradient-gymnasium',
            config=vars(args)
        )
        if wandb_run is not None:
            # Update args with sweep parameters if using W&B sweeps
            config = wandb.config
            for key in config.keys():
                if hasattr(args, key.replace('-', '_')):
                    setattr(args, key.replace('-', '_'), config[key])

            # Recompute environment dimensions AFTER sweep overrides (env may change)
            if args.env == 'LunarLander-v3':
                temp_env = gym.make(args.env, continuous=True)
            elif args.env == 'CarRacing-v3':
                temp_env = gym.make(args.env, continuous=True)
                temp_env = FrameSkip(temp_env, skip=4)
                temp_env = CarRacingActionWrapper(temp_env)
                temp_env = CarRacingPreprocessor(temp_env)
            else:
                temp_env = gym.make(args.env)
            
            # Detect if environment uses images (CNN needed)
            use_cnn = len(temp_env.observation_space.shape) > 1
            if use_cnn:
                input_channels = temp_env.observation_space.shape[0]
                state_size = temp_env.observation_space.shape[1]
            else:
                state_size = temp_env.observation_space.shape[0]
                input_channels = 1
            
            action_size = get_action_space_info(temp_env)
            temp_env.close()

    # Calculate entropy decay episodes from fraction (clamp between 0 and 1)
    entropy_decay = max(0.0, min(1.0, args.entropy_decay))
    entropy_decay_episodes = int(args.num_episodes * entropy_decay)

    # Display final configuration after potential sweep overrides
    print(f"\nEnvironment: {args.env}")
    print(f"Algorithm: {args.algorithm} | Device: {args.device}")
    print(f"State size: {state_size} | Action size: {action_size} | Action space: Continuous")
    print(f"Using CNN: {use_cnn}" + (f" | Input channels: {input_channels}" if use_cnn else ""))
    print(f"Episodes: {args.num_episodes} | Max steps: {args.max_steps} | Tests: {args.num_tests}")
    print(f"Batch size: {args.batch_size} | Hidden sizes: {args.hidden_sizes} | Reward scale: {args.reward_scale}")
    print(f"Learning Rate: {args.learning_rate} | Gamma: {args.gamma}")
    print(f"Render: {args.render} | Record video: {args.record_video} | W&B: {not args.no_wandb}")
    # PER summary
    print(f"Prioritized Replay: {args.use_per} "
          + (f"(alpha={args.per_alpha}, eps={args.per_eps}, beta={args.per_beta})" if args.use_per else ""))
    if args.algorithm == 'SAC':
        print(f"Buffer size: {args.buffer_size} | Tau: {args.tau} | Alpha: {args.alpha}")
    elif args.algorithm == 'TD3':
        print(f"Buffer size: {args.buffer_size} | Tau: {args.tau}")
        print(f"Exploration Noise: {args.exploration_noise} | Policy Noise: {args.policy_noise} | Noise Clip: {args.noise_clip} | Policy Delay: {args.policy_delay}")
    else:  # PPO
        print(f"PPO Epochs: {args.ppo_epochs} | Epsilon Clip: {args.epsilon_clip}")
        print(f"GAE Lambda: {args.gae_lambda} | Entropy Coef: {args.entropy_coef} -> 0 over {entropy_decay_episodes} episodes ({entropy_decay*100:.0f}%)")
        print(f"Max Grad Norm: {args.max_grad_norm} | Action Scale: {args.action_scale}")
    print()

    # Create agent after potential sweep overrides have been applied
    if args.algorithm == 'SAC':
        agent = SACAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            tau=args.tau,
            alpha=args.alpha,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
            use_cnn=use_cnn,
            input_channels=input_channels,
            reward_scale=args.reward_scale
        )
    elif args.algorithm == 'TD3':
        agent = TD3Agent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            tau=args.tau,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_delay=args.policy_delay,
            exploration_noise=args.exploration_noise,
            use_cnn=use_cnn,
            input_channels=input_channels,
            reward_scale=args.reward_scale,
            use_per=args.use_per,
            per_alpha=args.per_alpha,
            per_eps=args.per_eps,
            per_beta=args.per_beta
        )
    else:  # PPO
        agent = PPOAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            epsilon_clip=args.epsilon_clip,
            epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
            entropy_coef=args.entropy_coef,
            entropy_decay_episodes=entropy_decay_episodes,
            max_grad_norm=args.max_grad_norm,
            action_scale=args.action_scale,
            reward_scale=args.reward_scale,
            use_cnn=use_cnn,
            input_channels=input_channels
        )

    # Generate descriptive run name with key hyperparameters
    run_name = generate_run_name(
        algorithm=args.algorithm,
        env_name=args.env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        buffer_size=args.buffer_size if args.algorithm == 'SAC' else 0,
        batch_size=args.batch_size
    )

    if wandb_run is not None:
        wandb_run.name = run_name
    
    # Load model if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        agent.load(args.load_model)
    
    # Train or test
    if not args.test_only:
        print("Starting training...")
        agent = train_agent(
            agent=agent,
            env_name=args.env,
            run_name=run_name,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            log_interval=10,
            save_dir='models',
            use_wandb=use_wandb
        )
    
    # Test
    print("\nStarting testing...")
    test_results = test_agent(
        agent=agent,
        env_name=args.env,
        run_name=run_name,
        num_tests=args.num_tests,
        max_steps=args.max_steps,
        render=args.render,
        record_video=args.record_video,
        video_folder=f'videos',
        use_wandb=use_wandb,
        training_episodes=args.num_episodes if not args.test_only else 0
    )
    
    # Log test results to W&B
    if use_wandb:
        wandb.log({
            'test_mean_reward': test_results['mean_reward'],
            'test_std_reward': test_results['std_reward'],
            'test_mean_duration': test_results['mean_duration'],
            'test_std_duration': test_results['std_duration']
        })
        wandb.finish()
    
    print("\nTraining and testing complete!")


if __name__ == '__main__':
    main()