"""CMA-ES agent for Mountain Car (discrete and continuous)."""

from __future__ import annotations

from pathlib import Path

import cma
import gymnasium as gym
import numpy as np
import torch

from src.agents.nn_policy import build_policy


class CMAESAgent:
    def __init__(
        self,
        env,
        population_size: int = 50,
        sigma0: float = 0.5,
        hidden_sizes: list[int] = [64, 64],
    ) -> None:
        self._env = env
        self._net = build_policy(env, hidden_sizes)
        self._discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self._augment = env.observation_space.shape[0] == 4
        self._action_scale = float(env.action_space.high[0]) if not self._discrete else 1.0

        self._es = cma.CMAEvolutionStrategy(
            np.zeros(self._net.n_params),
            sigma0,
            {'popsize': population_size, 'verbose': -9},
        )
        self._best_weights: np.ndarray | None = None
        self._best_fitness: float = -np.inf

    def _eval_individual(self, weights: np.ndarray, max_steps: int) -> float:
        self._net.set_weights(weights)
        obs, _ = self._env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            action = self.predict(obs)
            obs, reward, terminated, truncated, _ = self._env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return total_reward

    def learn(self, total_timesteps: int) -> None:
        solutions = self._es.ask()
        fitnesses = [self._eval_individual(np.array(w), total_timesteps) for w in solutions]
        self._es.tell(solutions, [-f for f in fitnesses])

        best_idx = int(np.argmax(fitnesses))
        best_fitness = fitnesses[best_idx]
        if best_fitness > self._best_fitness:
            self._best_fitness = best_fitness
            self._best_weights = np.array(solutions[best_idx]).copy()

        if self._best_weights is not None:
            self._net.set_weights(self._best_weights)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int | np.ndarray:
        if self._best_weights is None:
            self._net.set_weights(np.zeros(self._net.n_params))
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
            best_weights=self._best_weights if self._best_weights is not None else np.array([]),
            best_fitness=np.array([self._best_fitness]),
            cma_mean=self._es.mean.copy(),
            cma_sigma=np.array([self._es.sigma]),
            cma_C=self._es.sm.C.copy(),
            cma_B=self._es.sm.B.copy(),
            cma_D=np.atleast_1d(self._es.sm.D).copy(),
            cma_count_tell=np.array([self._es.sm.count_tell], dtype=np.int64),
            cma_popsize=np.array([self._es.popsize], dtype=np.int64),
        )

    def load(self, model_path: str | Path, env=None) -> None:
        data = np.load(Path(model_path).with_suffix('.npz'), allow_pickle=False)
        w = data['best_weights']
        self._best_weights = w if w.size > 0 else None
        self._best_fitness = float(data['best_fitness'][0])
        if self._best_weights is not None:
            self._net.set_weights(self._best_weights)
        mean = data['cma_mean']
        sigma = float(data['cma_sigma'][0])
        popsize = int(data['cma_popsize'][0])
        self._es = cma.CMAEvolutionStrategy(mean, sigma, {'popsize': popsize, 'verbose': -9})
        self._es.sm.C = data['cma_C'].copy()
        self._es.sm.B = data['cma_B'].copy()
        self._es.sm.D = data['cma_D'].copy()
        self._es.sm.count_tell = int(data['cma_count_tell'][0])

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
