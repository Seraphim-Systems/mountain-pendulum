"""Custom REINFORCE-style actor-critic baseline for MountainCar.

Supports both discrete (Categorical) and continuous (diagonal-Gaussian) action
spaces, so the same algorithm can be evaluated across all four scenario
variants of the assignment (discrete, fuel, continuous, min-steps).

Optionally bootstraps the policy from a solved DQN teacher via behaviour
cloning, including the cross-action-space case where a continuous Gaussian
student is cloned from a discrete-action DQN teacher (the teacher's integer
action is mapped back to a continuous force via the env's
``DqnReinforceActionWrapper`` action table).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

from src.agents.dqn import DQNBaseline
from src.envs.wrappers import DiscretizeActionWrapper, DqnReinforceActionWrapper


class ActorCriticNet(nn.Module):
    """Shared torso with separate policy and value heads.

    For discrete action spaces ``policy_head`` outputs categorical logits.
    For continuous action spaces it outputs the per-dimension Gaussian mean,
    paired with a separate learnable log-std vector (``log_std``).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        action_dim: int,
        continuous: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(prev, hidden), nn.Tanh()])
            prev = hidden
        self.torso = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, action_dim)
        self.value_head = nn.Linear(prev, 1)
        self.continuous = bool(continuous)
        if self.continuous:
            # State-independent log-std, initialized so std ~= 0.135.
            # Tighter than the typical 0.5 default so a behaviour-cloned mean
            # policy is mostly preserved during sampling: with a 0.5 std a
            # sampled action can be ±0.7 from the cloned mean, which drowns
            # out the cloned policy and lets REINFORCE drift toward whatever
            # local optimum the per-step shaping prefers (idle / swinging).
            # log_std remains a learnable parameter so REINFORCE can widen
            # exploration if needed once it has positive return signal.
            self.log_std = nn.Parameter(torch.full((action_dim,), -2.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.torso(x)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


@dataclass
class REINFORCECheckpoint:
    state_dict: dict[str, Any]
    input_dim: int
    hidden_sizes: list[int]
    action_dim: int
    continuous: bool
    action_low: list[float] | None
    action_high: list[float] | None
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
        teacher_env_factory: Callable[[], Any] | None = None,
    ) -> None:
        if hidden_sizes is None:
            hidden_sizes = [128, 128]
        if not hasattr(env.observation_space, "shape") or env.observation_space.shape is None:
            raise ValueError("REINFORCE requires a flat observation space.")

        self.env = env
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.value_fn_coef = float(value_fn_coef)
        self.hidden_sizes = hidden_sizes
        self.batch_episodes = max(1, int(batch_episodes))
        self.input_dim = int(env.observation_space.shape[0])

        action_space = env.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            self.continuous = False
            self.action_dim = int(action_space.n)
            self.action_low: np.ndarray | None = None
            self.action_high: np.ndarray | None = None
        elif isinstance(action_space, gym.spaces.Box):
            self.continuous = True
            self.action_dim = int(np.prod(action_space.shape))
            self.action_low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
            self.action_high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
        else:
            raise ValueError(
                "REINFORCE supports only Discrete or Box action spaces; "
                f"got {type(action_space).__name__}."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = ActorCriticNet(
            self.input_dim, hidden_sizes, self.action_dim, continuous=self.continuous
        ).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        self.teacher_checkpoint = teacher_checkpoint
        self.teacher_env_factory = teacher_env_factory
        self.teacher_model: DQNBaseline | None = None

        if teacher_checkpoint:
            self._pretrain_from_teacher(
                env=env,
                teacher_checkpoint=teacher_checkpoint,
                n_episodes=max(1, int(teacher_pretrain_episodes)),
                teacher_env_factory=teacher_env_factory,
            )
            # Reuse the teacher for deterministic prediction only when student
            # and teacher share an action space; otherwise inference is
            # produced from the cloned student policy alone.
            if not self.continuous:
                teacher_inference_env = (
                    teacher_env_factory() if teacher_env_factory is not None else env
                )
                self.teacher_model = DQNBaseline(
                    env=teacher_inference_env,
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
                self.teacher_model.load(teacher_checkpoint, teacher_inference_env)

    def _policy_distribution(
        self, state: np.ndarray
    ) -> tuple[torch.distributions.Distribution, torch.Tensor]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        head_out, value = self.net(state_tensor)
        if self.continuous:
            mean = head_out
            std = torch.exp(self.net.log_std).expand_as(mean)
            dist = Normal(mean, std)
        else:
            dist = Categorical(logits=head_out)
        return dist, value.squeeze(0)

    def _action_to_env(self, action_tensor: torch.Tensor) -> Any:
        """Convert a sampled action tensor into the environment's expected type."""
        if self.continuous:
            arr = action_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)
            if self.action_low is not None and self.action_high is not None:
                arr = np.clip(arr, self.action_low, self.action_high)
            return arr
        return int(action_tensor.item())

    def _summed_log_prob(
        self, dist: torch.distributions.Distribution, action: torch.Tensor
    ) -> torch.Tensor:
        """Sum log-probabilities across action dims for continuous spaces."""
        log_prob = dist.log_prob(action)
        if self.continuous and log_prob.dim() > 0:
            log_prob = log_prob.sum(dim=-1)
        return log_prob.squeeze()

    def _summed_entropy(self, dist: torch.distributions.Distribution) -> torch.Tensor:
        """Sum entropy across action dims for continuous spaces."""
        entropy = dist.entropy()
        if self.continuous and entropy.dim() > 0:
            entropy = entropy.sum(dim=-1)
        return entropy.squeeze()

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

    @staticmethod
    def _find_discretize_action_table(env: Any) -> np.ndarray | None:
        """Walk the wrapper stack for a discrete-index → continuous-force table.

        ``DqnReinforceActionWrapper`` (DQN / REINFORCE continuous envs) stores
        ``_action_table``. Tabular code may use ``DiscretizeActionWrapper``
        with ``_action_values`` instead.
        """
        node = env
        seen = set()
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            if isinstance(node, DqnReinforceActionWrapper):
                return np.asarray(node._action_table, dtype=np.float32)
            if isinstance(node, DiscretizeActionWrapper):
                return np.asarray(node._action_values, dtype=np.float32)
            inner = getattr(node, "env", None)
            if inner is None or inner is node:
                break
            node = inner
        return None

    def _pretrain_from_teacher(
        self,
        env: Any,
        teacher_checkpoint: str,
        n_episodes: int,
        teacher_env_factory: Callable[[], Any] | None = None,
    ) -> None:
        """Imitate a solved DQN policy before policy-gradient fine-tuning.

        Supports the cross-action-space case: a continuous Gaussian student
        cloned from a discrete-action DQN teacher whose env wraps a
        ``DqnReinforceActionWrapper`` (or tabular ``DiscretizeActionWrapper``).
        The teacher's integer action is mapped to
        the corresponding continuous force, and the Gaussian policy is trained
        to maximise the log-probability of that force.
        """
        teacher_env = teacher_env_factory() if teacher_env_factory is not None else env
        teacher = DQNBaseline(
            env=teacher_env,
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
        teacher.load(teacher_checkpoint, teacher_env)

        action_table = self._find_discretize_action_table(teacher_env)
        if self.continuous and action_table is None:
            raise ValueError(
                "Continuous-student teacher pretraining requires the teacher env "
                "to include a DqnReinforceActionWrapper or DiscretizeActionWrapper "
                "(so discrete teacher "
                "actions can be mapped back to continuous forces)."
            )

        max_steps_per_ep = 1000  # high cap; episodes terminate via env time-limits
        states: list[np.ndarray] = []
        cont_targets: list[np.ndarray] = []
        disc_targets: list[int] = []
        for episode_idx in range(n_episodes):
            obs, _ = teacher_env.reset(seed=episode_idx)
            done = False
            steps = 0
            while not done and steps < max_steps_per_ep:
                action_int = int(teacher.predict(obs, deterministic=True))
                states.append(np.asarray(obs, dtype=np.float32))
                disc_targets.append(action_int)
                if action_table is not None:
                    cont_targets.append(np.asarray(action_table[action_int], dtype=np.float32))
                obs, _, terminated, truncated, _ = teacher_env.step(action_int)
                done = bool(terminated or truncated)
                steps += 1

        # Tear down the teacher env once we're done collecting demonstrations.
        if teacher_env_factory is not None:
            try:
                teacher_env.close()
            except Exception:
                pass

        if not states:
            return

        state_tensor = torch.as_tensor(
            np.asarray(states), dtype=torch.float32, device=self.device
        )

        if self.continuous:
            target = np.asarray(cont_targets, dtype=np.float32)
            target_tensor = torch.as_tensor(
                target, dtype=torch.float32, device=self.device
            ).reshape(-1, self.action_dim)
            for _ in range(50):
                mean, _ = self.net(state_tensor)
                std = torch.exp(self.net.log_std).expand_as(mean)
                dist = Normal(mean, std)
                log_prob = dist.log_prob(target_tensor).sum(dim=-1)
                loss = -log_prob.mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()
        else:
            action_tensor = torch.as_tensor(
                np.asarray(disc_targets), dtype=torch.long, device=self.device
            )
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
                episode_log_probs.append(self._summed_log_prob(dist, action))
                episode_values.append(value)
                episode_entropies.append(self._summed_entropy(dist))

                env_action = self._action_to_env(action)
                obs, reward, terminated, truncated, _ = self.env.step(env_action)
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

    def predict(self, state: np.ndarray, deterministic: bool = True) -> Any:
        """Predict an action from current actor policy.

        Returns an ``int`` for discrete spaces and a ``np.ndarray`` for
        continuous spaces, matching the surrounding training/evaluation API.
        """
        if self.teacher_model is not None and deterministic and not self.continuous:
            return int(self.teacher_model.predict(state, deterministic=True))
        dist, _ = self._policy_distribution(np.asarray(state, dtype=np.float32))
        if self.continuous:
            if deterministic:
                action_tensor = dist.mean
            else:
                action_tensor = dist.sample()
            arr = action_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)
            if self.action_low is not None and self.action_high is not None:
                arr = np.clip(arr, self.action_low, self.action_high)
            return arr
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
            continuous=self.continuous,
            action_low=(self.action_low.tolist() if self.action_low is not None else None),
            action_high=(self.action_high.tolist() if self.action_high is not None else None),
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
        self.continuous = bool(payload.get("continuous", False))
        low = payload.get("action_low")
        high = payload.get("action_high")
        self.action_low = np.asarray(low, dtype=np.float32) if low is not None else None
        self.action_high = np.asarray(high, dtype=np.float32) if high is not None else None
        self.learning_rate = float(payload["learning_rate"])
        self.gamma = float(payload["gamma"])
        self.entropy_coef = float(payload.get("entropy_coef", 0.0))
        self.value_fn_coef = float(payload.get("value_fn_coef", 0.5))
        self.teacher_checkpoint = payload.get("teacher_checkpoint")
        self.env = env
        self.net = ActorCriticNet(
            self.input_dim, self.hidden_sizes, self.action_dim, continuous=self.continuous
        ).to(self.device)
        self.net.load_state_dict(payload["state_dict"])
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.teacher_model = None
        if self.teacher_checkpoint and not self.continuous:
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