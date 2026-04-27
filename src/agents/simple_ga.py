"""Simple Genetic Algorithm agent for Mountain Car (discrete and continuous)."""

from __future__ import annotations

import copy
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from src.agents.nn_policy import build_policy


class SimpleGAAgent:
    def __init__(
        self,
        env,
        population_size: int = 50,
        elite_frac: float = 0.2,
        mutation_std: float = 0.05,
        crossover_alpha: float = 0.5,
        hidden_sizes: list[int] = [64, 64],
    ) -> None:
        self._env = env
        self._net = build_policy(env, hidden_sizes)
        self._discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self._augment = env.observation_space.shape[0] == 4
        self._action_scale = float(env.action_space.high[0]) if not self._discrete else 1.0

        self._population_size = population_size
        self._elite_frac = elite_frac
        self._mutation_std = mutation_std
        self._crossover_alpha = crossover_alpha
        self._hidden_sizes = hidden_sizes
        self._rng = np.random.default_rng()
        self._envs: list | None = None

        self._pop = self._rng.standard_normal((population_size, self._net.n_params)) * 0.1
        self._best_weights: np.ndarray | None = None
        self._best_fitness: float = -np.inf

    def _get_envs(self, n: int) -> list:
        if self._envs is None or len(self._envs) != n:
            self._envs = [copy.deepcopy(self._env) for _ in range(n)]
        return self._envs

    def _eval_population(self, weights_matrix: np.ndarray, max_steps: int) -> list[float]:
        N = len(weights_matrix)
        envs = self._get_envs(N)
        obs = np.array([env.reset()[0] for env in envs], dtype=np.float32)
        total_rewards = np.zeros(N)
        active = np.ones(N, dtype=bool)

        for _ in range(max_steps):
            if not active.any():
                break
            idx = np.where(active)[0]
            logits = self._net.batched_forward(obs[idx], weights_matrix[idx])
            if self._discrete:
                actions = np.argmax(logits, axis=1)
            else:
                actions = (np.tanh(logits) * self._action_scale).astype(np.float32)
            for j, i in enumerate(idx):
                act = int(actions[j]) if self._discrete else actions[j]
                next_obs, reward, terminated, truncated, _ = envs[i].step(act)
                total_rewards[i] += float(reward)
                obs[i] = next_obs
                if terminated or truncated:
                    active[i] = False

        return total_rewards.tolist()

    def learn(self, total_timesteps: int) -> None:
        fitnesses = self._eval_population(self._pop, total_timesteps)
        ranked = np.argsort(fitnesses)[::-1]
        n_elite = max(1, int(len(self._pop) * self._elite_frac))

        elites = self._pop[ranked[:n_elite]]
        new_pop = list(elites)

        n_params = self._net.n_params
        for _ in range(self._population_size - n_elite):
            idx_a, idx_b = self._rng.choice(n_elite, size=2, replace=n_elite < 2)
            parent_a, parent_b = elites[idx_a], elites[idx_b]
            child = self._crossover_alpha * parent_a + (1 - self._crossover_alpha) * parent_b
            child = child + self._rng.standard_normal(n_params) * self._mutation_std
            new_pop.append(child)

        self._pop = np.array(new_pop)

        best_fitness = max(fitnesses)
        if best_fitness > self._best_fitness:
            self._best_fitness = best_fitness
            self._best_weights = elites[0].copy()

        self._net.set_weights(self._best_weights)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int | np.ndarray:
        if self._best_weights is None:
            self._net.set_weights(self._pop[0])
        tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        with torch.no_grad():
            logits = self._net(tensor)
        if self._discrete:
            return int(np.argmax(logits.numpy()))
        return (np.tanh(logits.numpy()) * self._action_scale).astype(np.float32).reshape(-1)

    def save(self, model_path: str | Path) -> None:
        path = Path(model_path).with_suffix('.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            population=self._pop,
            best_weights=self._best_weights if self._best_weights is not None else np.array([]),
            best_fitness=np.array([self._best_fitness]),
        )

    def load(self, model_path: str | Path, env=None) -> None:
        data = np.load(Path(model_path).with_suffix('.npz'), allow_pickle=False)
        self._pop = data['population']
        w = data['best_weights']
        self._best_weights = w if w.size > 0 else None
        self._best_fitness = float(data['best_fitness'][0])
        if self._best_weights is not None:
            self._net.set_weights(self._best_weights)

    @property
    def policy_table(self) -> np.ndarray | None:
        if not self._discrete:
            return None
        from src.envs.state_utils import augment_state
        pos_bins = np.linspace(-1.2, 0.6, 24)
        vel_bins = np.linspace(-0.07, 0.07, 24)
        table = np.zeros((24, 24), dtype=np.int32)
        with torch.no_grad():
            for i, pos in enumerate(pos_bins):
                for j, vel in enumerate(vel_bins):
                    obs = np.array([pos, vel], dtype=np.float32)
                    if self._augment:
                        obs = augment_state(obs)
                    logits = self._net(torch.from_numpy(obs))
                    table[i, j] = int(np.argmax(logits.numpy()))
        return table
