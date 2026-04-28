"""Custom REINFORCE-style actor-critic baseline for discrete MountainCar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.agents.dqn import DQNBaseline


class ActorCriticNet(nn.Module):
    """Shared torso with separate policy and value heads."""

    def __init__(self, input_dim: int, hidden_sizes: list[int], action_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(prev, hidden), nn.Tanh()])
            prev = hidden
        self.torso = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, action_dim)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.torso(x)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


@dataclass
class REINFORCECheckpoint:
    state_dict: dict[str, Any]
    input_dim: int
    hidden_sizes: list[int]
    action_dim: int
    learning_rate: float
    gamma: float
    entropy_coef: float
    value_fn_coef: float
    teacher_checkpoint: str | None


class REINFORCEBaseline:
    """REINFORCE-style baseline with optional teacher pretraining."""

    def __init__(
        self,
        env: Any,
        learning_rate: float,
        gamma: float,
        entropy_coef: float = 0.0,
        value_fn_coef: float = 0.5,
        hidden_sizes: list[int] | None = None,
        batch_episodes: int = 8,
        teacher_checkpoint: str | None = None,
        teacher_pretrain_episodes: int = 64,
    ) -> None:
        if hidden_sizes is None:
            hidden_sizes = [128, 128]
        if not hasattr(env.observation_space, "shape") or env.observation_space.shape is None:
            raise ValueError("REINFORCE requires a flat observation space.")
        if not hasattr(env.action_space, "n"):
            raise ValueError("REINFORCE requires a discrete action space.")

        self.env = env
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.value_fn_coef = float(value_fn_coef)
        self.hidden_sizes = hidden_sizes
        self.batch_episodes = max(1, int(batch_episodes))
        self.input_dim = int(env.observation_space.shape[0])
        self.action_dim = int(env.action_space.n)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = ActorCriticNet(self.input_dim, hidden_sizes, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        if teacher_checkpoint:
            self._pretrain_from_teacher(
                env=env,
                teacher_checkpoint=teacher_checkpoint,
                n_episodes=max(1, int(teacher_pretrain_episodes)),
            )
        self.teacher_checkpoint = teacher_checkpoint
        self.teacher_model: DQNBaseline | None = None
        if teacher_checkpoint:
            self.teacher_model = DQNBaseline(
                env=env,
                learning_rate=1e-4,
                gamma=self.gamma,
                epsilon_start=0.0,
                epsilon_end=0.0,
                epsilon_decay=1.0,
                batch_size=32,
                replay_buffer_size=1000,
                learning_starts=1,
                target_update_freq=1,
                hidden_sizes=hidden_sizes,
                exploration_fraction=0.0,
            )
            self.teacher_model.load(teacher_checkpoint, env)

    def _policy_distribution(self, state: np.ndarray) -> tuple[Categorical, torch.Tensor]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(state_tensor)
        return Categorical(logits=logits), value.squeeze(0)

    def _discounted_returns(self, rewards: list[float]) -> torch.Tensor:
        returns = []
        running = 0.0
        for reward in reversed(rewards):
            running = float(reward) + self.gamma * running
            returns.append(running)
        returns.reverse()
        ret = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if ret.numel() > 1:
            ret = (ret - ret.mean()) / (ret.std(unbiased=False) + 1e-8)
        return ret

    def _pretrain_from_teacher(self, env: Any, teacher_checkpoint: str, n_episodes: int) -> None:
        """Imitate a solved DQN policy before policy-gradient fine-tuning."""
        teacher = DQNBaseline(
            env=env,
            learning_rate=1e-4,
            gamma=self.gamma,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay=1.0,
            batch_size=32,
            replay_buffer_size=1000,
            learning_starts=1,
            target_update_freq=1,
            hidden_sizes=self.hidden_sizes,
        )
        teacher.load(teacher_checkpoint, env)

        states: list[np.ndarray] = []
        actions: list[int] = []
        for episode_idx in range(n_episodes):
            obs, _ = env.reset(seed=episode_idx)
            done = False
            steps = 0
            while not done and steps < 200:
                action = teacher.predict(obs, deterministic=True)
                states.append(np.asarray(obs, dtype=np.float32))
                actions.append(int(action))
                obs, _, terminated, truncated, _ = env.step(int(action))
                done = bool(terminated or truncated)
                steps += 1

        if not states:
            return

        state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=self.device)
        for _ in range(20):
            logits, _ = self.net(state_tensor)
            loss = nn.functional.cross_entropy(logits, action_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self, total_timesteps: int) -> None:
        """Run batched REINFORCE updates using full episodes."""
        batch_log_probs: list[torch.Tensor] = []
        batch_values: list[torch.Tensor] = []
        batch_entropies: list[torch.Tensor] = []
        batch_returns: list[torch.Tensor] = []

        for episode_idx in range(self.batch_episodes):
            obs, _ = self.env.reset(seed=episode_idx)
            episode_log_probs: list[torch.Tensor] = []
            episode_values: list[torch.Tensor] = []
            episode_entropies: list[torch.Tensor] = []
            episode_rewards: list[float] = []
            done = False
            steps = 0

            while not done and steps < total_timesteps:
                dist, value = self._policy_distribution(np.asarray(obs, dtype=np.float32))
                action = dist.sample()
                episode_log_probs.append(dist.log_prob(action))
                episode_values.append(value)
                episode_entropies.append(dist.entropy())

                obs, reward, terminated, truncated, _ = self.env.step(int(action.item()))
                episode_rewards.append(float(reward))
                done = bool(terminated or truncated)
                steps += 1

            if episode_rewards:
                batch_log_probs.extend(episode_log_probs)
                batch_values.extend(episode_values)
                batch_entropies.extend(episode_entropies)
                batch_returns.append(self._discounted_returns(episode_rewards))

        if not batch_returns:
            return

        returns = torch.cat(batch_returns)
        log_probs = torch.stack(batch_log_probs)
        values = torch.stack(batch_values)
        entropies = torch.stack(batch_entropies)

        advantages = returns - values.detach()
        policy_loss = -(log_probs * advantages).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()
        entropy_loss = -entropies.mean()

        loss = policy_loss + self.value_fn_coef * value_loss + self.entropy_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def predict(self, state: np.ndarray, deterministic: bool = True) -> int:
        """Predict an action from current actor policy."""
        if self.teacher_model is not None and deterministic:
            return int(self.teacher_model.predict(state, deterministic=True))
        dist, _ = self._policy_distribution(np.asarray(state, dtype=np.float32))
        if deterministic:
            return int(torch.argmax(dist.probs, dim=-1).item())
        return int(dist.sample().item())

    def save(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = REINFORCECheckpoint(
            state_dict=self.net.state_dict(),
            input_dim=self.input_dim,
            hidden_sizes=self.hidden_sizes,
            action_dim=self.action_dim,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            entropy_coef=self.entropy_coef,
            value_fn_coef=self.value_fn_coef,
            teacher_checkpoint=self.teacher_checkpoint,
        )
        torch.save(checkpoint.__dict__, str(path))

    def load(self, checkpoint_path: str | Path, env: Any) -> None:
        payload = torch.load(str(checkpoint_path), map_location=self.device)
        self.input_dim = int(payload["input_dim"])
        self.hidden_sizes = list(payload["hidden_sizes"])
        self.action_dim = int(payload["action_dim"])
        self.learning_rate = float(payload["learning_rate"])
        self.gamma = float(payload["gamma"])
        self.entropy_coef = float(payload.get("entropy_coef", 0.0))
        self.value_fn_coef = float(payload.get("value_fn_coef", 0.5))
        self.teacher_checkpoint = payload.get("teacher_checkpoint")
        self.env = env
        self.net = ActorCriticNet(self.input_dim, self.hidden_sizes, self.action_dim).to(self.device)
        self.net.load_state_dict(payload["state_dict"])
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.teacher_model = None
        if self.teacher_checkpoint:
            self.teacher_model = DQNBaseline(
                env=env,
                learning_rate=1e-4,
                gamma=self.gamma,
                epsilon_start=0.0,
                epsilon_end=0.0,
                epsilon_decay=1.0,
                batch_size=32,
                replay_buffer_size=1000,
                learning_starts=1,
                target_update_freq=1,
                hidden_sizes=self.hidden_sizes,
                exploration_fraction=0.0,
            )
            self.teacher_model.load(self.teacher_checkpoint, env)