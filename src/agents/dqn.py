"""Stable-Baselines3 DQN baseline adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import DQN


class DQNBaseline:
    """DQN baseline for discrete-action MountainCar with function approximation."""

    def __init__(
        self,
        env: Any,
        learning_rate: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        batch_size: int,
        replay_buffer_size: int,
        learning_starts: int,
        target_update_freq: int,
        hidden_sizes: list[int],
        exploration_fraction: float | None = None,
        total_timesteps_budget: int | None = None,
    ) -> None:
        if exploration_fraction is None:
            exploration_fraction = min(1.0, max(1e-4, 1.0 - epsilon_decay))
        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            gamma=gamma,
            exploration_initial_eps=epsilon_start,
            exploration_final_eps=epsilon_end,
            exploration_fraction=exploration_fraction,
            batch_size=batch_size,
            buffer_size=replay_buffer_size,
            learning_starts=learning_starts,
            target_update_interval=target_update_freq,
            policy_kwargs={"net_arch": hidden_sizes},
            verbose=0,
        )
        # When ``learn()`` is invoked repeatedly with a small per-call budget
        # (the project's per-episode loop pattern), SB3 unconditionally resets
        # ``self._total_timesteps`` at the start of every ``learn()`` to
        # ``per_call_budget + num_timesteps``. That makes the exploration
        # schedule decay to ``exploration_final_eps`` after a single call and
        # effectively disables exploration. Patching ``_setup_learn`` on the
        # model instance lets us keep the schedule denominator at the full
        # training horizon for every call.
        self._total_budget = (
            int(total_timesteps_budget) if total_timesteps_budget is not None else None
        )
        if self._total_budget is not None:
            self._install_total_budget_patch()

    def _install_total_budget_patch(self) -> None:
        """Override ``_setup_learn`` so ``_total_timesteps`` keeps the full horizon.

        SB3's exploration schedule reads ``self._total_timesteps`` to compute
        ``progress_remaining``. With our per-episode learn loop the value
        defaults to one episode's worth of steps after every call, so the
        schedule jumps to ``exploration_final_eps`` immediately. The wrapper
        below restores the full horizon after SB3's setup runs.

        The patched closure captures ``self.model``, which can hold a
        ``SubprocVecEnv``; that confuses cloudpickle during ``model.save``.
        ``save()`` therefore temporarily uninstalls the patch.
        """
        self._original_setup_learn = self.model._setup_learn
        budget = int(self._total_budget) if self._total_budget is not None else 0
        original_setup = self._original_setup_learn

        def patched_setup(*args: Any, **kwargs: Any):
            result = original_setup(*args, **kwargs)
            self.model._total_timesteps = budget
            return result

        self.model._setup_learn = patched_setup  # type: ignore[assignment]

    def learn(self, total_timesteps: int) -> None:
        """Run DQN updates for the provided number of steps."""
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

    def predict(self, state: np.ndarray, deterministic: bool = True) -> int:
        """Predict an action from current policy network."""
        action, _ = self.model.predict(state, deterministic=deterministic)
        return int(action)

    def save(self, model_path: str | Path) -> None:
        """Save DQN model checkpoint.

        Temporarily removes the exploration-budget patch around the SB3 save
        call. ``_setup_learn`` is normally a class method and never lives in
        ``model.__dict__``; our patch stores it as an *instance* attribute,
        which SB3's save then tries to cloudpickle. The closure references
        ``self.model`` (and through it the ``SubprocVecEnv``'s unpicklable
        multiprocessing handles), so we delete the instance attribute for
        the duration of save and reinstall it afterward.
        """
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        had_patch = "_setup_learn" in self.model.__dict__
        if had_patch:
            del self.model.__dict__["_setup_learn"]
        try:
            self.model.save(str(path))
        finally:
            if had_patch:
                self._install_total_budget_patch()

    def load(self, model_path: str | Path, env: Any) -> None:
        """Load DQN model checkpoint bound to an environment."""
        self.model = DQN.load(str(model_path), env=env)
        if self._total_budget is not None:
            self._install_total_budget_patch()

    @property
    def exploration_rate(self) -> float:
        """Return the current epsilon-like exploration rate."""
        return float(self.model.exploration_rate)
