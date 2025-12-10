"""
Gymnasium Environment Runner for SAC, PPO, and TD3
Updated: improved CarRacing handling for PPO (no aggressive frame-skip),
video recording, and correct action scaling.
"""

import gymnasium as gym
import numpy as np
import os
import argparse
from datetime import datetime
from gymnasium.wrappers import RecordVideo
import cv2
import wandb

from models import SACAgent, PPOAgent, TD3Agent


class CarRacingPreprocessor(gym.ObservationWrapper):
    """Preprocessing wrapper for CarRacing: grayscale, resize, and frame stack."""

    def __init__(self, env, img_size=84, stack_frames=4):
        super().__init__(env)
        self.img_size = img_size
        self.stack_frames = stack_frames
        self.frames = []

        # Update observation space: (C, H, W) with uint8 [0,255]
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

        # Stack frames (C, H, W)
        return np.stack(self.frames, axis=0).astype(np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = []
        return self.observation(obs), info


class CarRacingActionWrapper(gym.ActionWrapper):
    """
    Rescale agent actions from [-1, 1] to the valid range for CarRacing-v3.
    Agent outputs: [Steer, Gas, Brake] all in [-1, 1]
    Environment expects:
        Steer: [-1, 1]
        Gas: [0, 1]
        Brake: [0, 1]
    """

    def action(self, action):
        steer = float(action[0])

        # Gas: [-1,1] -> [0,1]
        gas = float((action[1] + 1.0) / 2.0)

        # Brake: [-1,1] -> [0,1]
        brake = float((action[2] + 1.0) / 2.0)

        # Clip just in case
        steer = np.clip(steer, -1.0, 1.0)
        gas = np.clip(gas, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        return np.array([steer, gas, brake], dtype=np.float32)


class FrameSkip(gym.Wrapper):
    """Skip frames: repeat the given action for `skip` frames and accumulate reward."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        info = {}
        terminated = False
        truncated = False
        obs = None

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


def generate_run_name(algorithm, env_name, learning_rate, gamma, buffer_size, batch_size):
    """
    Generate a descriptive name for the run that includes key hyperparameters.
    """
    env_short = env_name.split('-')[0]
    buf_k = (buffer_size // 1000) if buffer_size else 0
    return f"{algorithm}_{env_short}_lr{learning_rate}_g{gamma}_buf{buf_k}k_bs{batch_size}"


def create_environment(env_name, algorithm=None, render_mode=None, video_folder=None, episode_trigger=None, name_prefix=None):
    """
    Create a Gymnasium environment with appropriate wrappers.
    `algorithm` (string) is used to choose different wrappers for PPO vs off-policy algos.
    """
    if env_name == 'LunarLander-v3':
        env = gym.make(env_name, continuous=True, render_mode=render_mode if video_folder else None)
    elif env_name == 'CarRacing-v3':
        car_render_mode = render_mode if render_mode else 'rgb_array'
        env = gym.make(env_name, continuous=True, render_mode=car_render_mode)

        # For PPO, prefer skip=1 (no aggressive frame skipping). Off-policy (SAC/TD3) can use skip=4.
        skip = 1 if (algorithm is not None and 'PPO' in algorithm) else 4
        if skip > 1:
            env = FrameSkip(env, skip=skip)

        # Action wrapper fixes [-1,1] -> [0,1] mapping for gas/brake
        env = CarRacingActionWrapper(env)

        # Preprocessing: grayscale, resize, stack frames
        env = CarRacingPreprocessor(env)

    else:
        env = gym.make(env_name, render_mode=render_mode if video_folder else None)

    if video_folder:
        if name_prefix is None:
            name_prefix = f"{env_name}-rl-video"
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=episode_trigger if episode_trigger else (lambda x: True),
            name_prefix=name_prefix
        )

    return env


def get_action_space_info(env):
    """Get info about the action space: (size, is_discrete)."""
    if isinstance(env.action_space, gym.spaces.Discrete):
        return env.action_space.n, True
    else:
        return env.action_space.shape[0], False


def train_agent(
    agent,
    env_name,
    run_name,
    num_episodes=1000,
    max_steps=500,
    log_interval=10,
    save_dir='models',
    use_wandb=False
):
    """
    Train the agent on the specified environment.
    `agent` is expected to be an instance of SACAgent, PPOAgent, or TD3Agent.
    This function will choose environment wrappers suitable for the agent class.
    """
    algo_name = type(agent).__name__ if agent is not None else None
    env = create_environment(env_name, algorithm=algo_name)

    os.makedirs(save_dir, exist_ok=True)
    episode_rewards = []
    episode_losses = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_loss = []

        if episode == 0:
            print(f"Episode {episode + 1} started, collecting experiences...")

        for step in range(max_steps):
            action = agent.select_action(state)  # agent must return actions in [-1,1] and manage internal log_probs for PPO

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # store transition; PPO's store_transition reads agent.last_log_prob internally when available
            agent.store_transition(state, action, reward, next_state, terminated)

            # If agent uses replay buffer (SAC/TD3), train every step
            if hasattr(agent, 'memory'):
                loss = agent.train()
                if loss is not None:
                    episode_loss.append(loss)
                    if episode == 0 and len(episode_loss) == 1:
                        print(f"Training started at step {step + 1} (replay buffer ready)")

            episode_reward += reward
            state = next_state

            if episode == 0 and (step + 1) % 50 == 0:
                print(f"  Step {step + 1}/{max_steps}, reward so far: {episode_reward:.2f}")

            if done:
                break

        # For on-policy algorithms like PPO, train at episode end
        if not hasattr(agent, 'memory'):
            loss = agent.train()
            if loss is not None:
                episode_loss.append(loss)

        episode_rewards.append(episode_reward)
        avg_loss = float(np.mean(episode_loss)) if episode_loss else 0.0
        episode_losses.append(avg_loss)

        # Log to wandb if enabled
        if use_wandb:
            entropy_or_alpha = agent.get_entropy_coef() if hasattr(agent, 'get_entropy_coef') else 0.0
            wandb.log({
                'episode': episode + 1,
                'episode_reward': episode_reward,
                'episode_loss': avg_loss,
                'entropy_or_alpha': entropy_or_alpha
            })

        if (episode + 1) % log_interval == 0:
            avg_reward = float(np.mean(episode_rewards[-log_interval:]))
            avg_loss = float(np.mean(episode_losses[-log_interval:]))
            entropy_or_alpha = agent.get_entropy_coef() if hasattr(agent, 'get_entropy_coef') else 0.0
            print(f"Episode {episode + 1}/{num_episodes} - Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}, Entropy/Alpha: {entropy_or_alpha:.4f}")
            
            if use_wandb:
                wandb.log({
                    'avg_reward': avg_reward,
                    'avg_loss': avg_loss,
                })

        # Optionally save intermediate checkpoints
        # if (episode + 1) % (log_interval * 5) == 0:
        #     ndir = os.path.join(save_dir, f"{run_name}_ep{episode+1}.pt")
        #     agent.save(ndir)
        #     print(f"Saved checkpoint: {ndir}")

    # Save final model
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
    """Test the trained agent and optionally record videos."""
    algo_name = type(agent).__name__ if agent is not None else None

    if record_video:
        os.makedirs(video_folder, exist_ok=True)
        env = create_environment(
            env_name,
            algorithm=algo_name,
            render_mode='rgb_array',
            video_folder=video_folder,
            episode_trigger=lambda x: x % 10 == 0,
            name_prefix=run_name
        )
    else:
        env = create_environment(env_name, algorithm=algo_name, render_mode='human' if render else None)

    test_rewards = []
    test_durations = []

    for test_episode in range(num_tests):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_duration = 0

        for step in range(max_steps):
            action = agent.select_action(state, epsilon=0.0)  # deterministic/eval mode if supported
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_duration += 1
            state = next_state

            if done:
                break

        test_rewards.append(episode_reward)
        test_durations.append(episode_duration)

        if (test_episode + 1) % 10 == 0:
            print(f"Test Episode {test_episode + 1}/{num_tests} - Reward: {episode_reward:.2f}, Duration: {episode_duration}")

    env.close()

    results = {
        'mean_reward': float(np.mean(test_rewards)),
        'std_reward': float(np.std(test_rewards)),
        'mean_duration': float(np.mean(test_durations)),
        'std_duration': float(np.std(test_durations)),
        'min_reward': float(np.min(test_rewards)),
        'max_reward': float(np.max(test_rewards))
    }

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Duration: {results['mean_duration']:.2f} ± {results['std_duration']:.2f}")
    print(f"Reward Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
    
    # Log test results to wandb
    if use_wandb:
        wandb.log({
            'test_mean_reward': results['mean_reward'],
            'test_std_reward': results['std_reward'],
            'test_mean_duration': results['mean_duration'],
            'test_std_duration': results['std_duration'],
            'test_min_reward': results['min_reward'],
            'test_max_reward': results['max_reward']
        })
    print("=" * 50 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train SAC/PPO/TD3 on Gymnasium environments')

    parser.add_argument('--env', type=str, default='CarRacing-v3', choices=['LunarLander-v3', 'CarRacing-v3'])
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['SAC', 'PPO', 'TD3'])
    parser.add_argument('--num-episodes', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--num-tests', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-sizes', nargs='+', type=int, default=[256, 256])
    parser.add_argument('--ppo-epochs', type=int, default=10)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--entropy-decay', type=float, default=0.5)
    parser.add_argument('--epsilon-clip', type=float, default=0.2,
                        help='PPO clipping epsilon for policy updates')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='Max gradient norm for PPO gradient clipping')
    parser.add_argument('--device', type=str, default='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--load-model', type=str, default=None)
    parser.add_argument('--record-video', action='store_true')
    parser.add_argument('--render', action='store_true')

    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # For dimension detection choose env wrappers consistent with chosen algorithm
    # Use create_environment for all envs to ensure consistency with training
    temp_env = create_environment(args.env, algorithm=args.algorithm)

    use_cnn = len(temp_env.observation_space.shape) > 1
    if use_cnn:
        input_channels = temp_env.observation_space.shape[0]
        state_size = temp_env.observation_space.shape[1]
    else:
        state_size = temp_env.observation_space.shape[0]
        input_channels = 1

    action_size, is_discrete = get_action_space_info(temp_env)
    temp_env.close()

    print(f"Environment: {args.env}")
    print(f"State size: {state_size} | Channels: {input_channels} | Action size: {action_size}")
    print(f"Algorithm: {args.algorithm} | Device: {args.device} | Use CNN: {use_cnn}")

    if args.algorithm == 'SAC':
        agent = SACAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            tau=0.005,
            alpha=0.2,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
            discrete=is_discrete,
            use_cnn=use_cnn,
            input_channels=input_channels,
            reward_scale=1.0
        )
    elif args.algorithm == 'TD3':
        agent = TD3Agent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            tau=0.005,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_delay=2,
            exploration_noise=0.1,
            use_cnn=use_cnn,
            input_channels=input_channels,
            reward_scale=1.0
        )
    else:  # PPO
        entropy_decay_episodes = int(args.num_episodes * max(0.0, min(1.0, args.entropy_decay)))
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
            discrete=is_discrete,
            entropy_coef=args.entropy_coef,
            entropy_decay_episodes=entropy_decay_episodes,
            max_grad_norm=args.max_grad_norm,
            action_scale=1.0,
            reward_scale=0.1,
            use_cnn=use_cnn,
            input_channels=input_channels
        )

    run_name = generate_run_name(
        algorithm=args.algorithm,
        env_name=args.env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        buffer_size=(args.buffer_size if args.algorithm in ['SAC', 'TD3'] else 0),
        batch_size=args.batch_size
    )

    if args.load_model:
        print(f"Loading model from {args.load_model}")
        agent.load(args.load_model)

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
            use_wandb=not args.no_wandb
        )

    print("\nStarting testing...")
    test_results = test_agent(
        agent=agent,
        env_name=args.env,
        run_name=run_name,
        num_tests=args.num_tests,
        max_steps=args.max_steps,
        render=args.render,
        record_video=args.record_video,
        video_folder='videos',
        use_wandb=not args.no_wandb,
        training_episodes=args.num_episodes if not args.test_only else 0
    )

    print("\nTraining and testing complete!")


if __name__ == '__main__':
    main()
