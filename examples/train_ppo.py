"""An example of training PPO in Voxel Gym 2D."""
import os
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch import nn

# Create log dir
LOG_DIR = "./logs/ppo_onestep/"
os.makedirs(LOG_DIR, exist_ok=True)


# feature extractor
class SimpleCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# callbacks
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq, log_dir, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.chckpoint_path = os.path.join(log_dir, "checkpoint_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                _mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f}")
                    print(f"Last mean reward per episode: {_mean_reward:.2f}")

                # New best model, you could save the agent here
                if _mean_reward > self.best_mean_reward:
                    self.best_mean_reward = _mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model at {x[-1]} timesteps")
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
                else:
                    if self.verbose > 0:
                        print(f"Saving checkpoint model at {x[-1]} timesteps")
                        print(f"Saving checkpoint model to {self.chckpoint_path}.zip")
                    self.model.save(self.chckpoint_path)

        return True


# scheduler
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


if __name__ == "__main__":
    NUM_CPU = 6  # Number of processes to use
    # Create the vectorized environment
    env = make_vec_env(
        env_id="voxelgym2D:onestep-v0",
        n_envs=NUM_CPU,
        seed=1327455,
        monitor_dir=LOG_DIR,
        env_kwargs={
            "mapfile": "600x600.npy",
            "view_size": 21,
            "image_size": 42,
            "max_collisions": 0,
            "max_steps": 60,
            "discrete_actions": True,
            "render_mode": "None",
        },
        vec_env_cls=SubprocVecEnv,
    )

    policy_kwargs = dict(
        normalize_images=True,
        features_extractor_class=SimpleCNN,
        features_extractor_kwargs=dict(features_dim=2048),
        net_arch=[dict(vf=[512, 256], pi=[512, 256])],
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(2.5e-4),
        gamma=0.9,
        n_steps=256,
        clip_range=linear_schedule(0.1),
        n_epochs=5,
        batch_size=256,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log="tb_logs/ppo_onestep/",
        target_kl=0.4,
    )

    # Create eval env
    eval_env = DummyVecEnv(
        [
            lambda: Monitor(
                gym.make(
                    "voxelgym2D:onestep-v0",
                    mapfile="600x600.npy",
                    view_size=21,
                    image_size=42,
                    max_collisions=0,
                    max_steps=60,
                    discrete_actions=True,
                    render_mode="None",
                ),
                filename=os.path.join(LOG_DIR, "eval"),
            )
        ]
    )

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=os.path.join(LOG_DIR, "eval"),
        eval_freq=10000,
        n_eval_episodes=50,
        deterministic=True,
        render=False,
    )

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50, warn=False)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    # Create Callback
    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=LOG_DIR)

    TOTAL_TIME_STEPS = 10000000

    model.learn(
        total_timesteps=TOTAL_TIME_STEPS,
        callback=[auto_save_callback, eval_callback],
        progress_bar=True,
    )

    model.save(os.path.join(LOG_DIR, "ppo_saved"))

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50, warn=False)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
