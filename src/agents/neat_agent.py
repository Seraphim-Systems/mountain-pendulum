"""NEAT agent for Mountain Car (discrete and continuous)."""

from __future__ import annotations

import configparser
import json
import tempfile
from math import inf
from pathlib import Path

import gymnasium as gym
import neat
import numpy as np


_DISCRETE_TEMPLATE = Path(__file__).parent.parent.parent / "configs" / "neat_discrete_template.ini"
_CONTINUOUS_TEMPLATE = Path(__file__).parent.parent.parent / "configs" / "neat_continuous_template.ini"


class NeatAgent:
    def __init__(
        self,
        env,
        pop_size: int = 50,
        fitness_threshold: float = 90.0,
        hidden_sizes=None,  # accepted but ignored; NEAT grows its own topology
    ) -> None:
        self._env = env
        self._discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self._augment = env.observation_space.shape[0] == 4
        self._action_scale = float(env.action_space.high[0]) if not self._discrete else 1.0

        num_inputs = env.observation_space.shape[0]
        num_outputs = env.action_space.n if self._discrete else env.action_space.shape[0]

        template = _DISCRETE_TEMPLATE if self._discrete else _CONTINUOUS_TEMPLATE
        self._config = self._build_config(template, num_inputs, num_outputs, pop_size, fitness_threshold)

        self._population = neat.Population(self._config)
        self._best_genome = None
        self._best_fitness: float = -inf

    def _build_config(
        self,
        template_path: Path,
        num_inputs: int,
        num_outputs: int,
        pop_size: int,
        fitness_threshold: float,
    ) -> neat.Config:
        parser = configparser.ConfigParser()
        parser.read(str(template_path))
        parser["NEAT"]["pop_size"] = str(pop_size)
        parser["NEAT"]["fitness_threshold"] = str(fitness_threshold)
        parser["DefaultGenome"]["num_inputs"] = str(num_inputs)
        parser["DefaultGenome"]["num_outputs"] = str(num_outputs)

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False)
        parser.write(tmp)
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            tmp_path,
        )
        Path(tmp_path).unlink(missing_ok=True)
        return config

    def _activate(self, genome, obs: np.ndarray):
        net = neat.nn.FeedForwardNetwork.create(genome, self._config)
        return net.activate(obs.tolist())

    def _eval_genome(self, genome, max_steps: int) -> float:
        obs, _ = self._env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            output = self._activate(genome, obs)
            if self._discrete:
                action = int(np.argmax(output))
            else:
                action = np.array(
                    [np.tanh(v) * self._action_scale for v in output], dtype=np.float32
                )
            obs, reward, terminated, truncated, _ = self._env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return total_reward

    def learn(self, total_timesteps: int) -> None:
        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = self._eval_genome(genome, total_timesteps)

        self._population.run(eval_genomes, 1)

        best = max(
            self._population.population.values(),
            key=lambda g: g.fitness if g.fitness is not None else -inf,
        )
        if best.fitness is not None and best.fitness > self._best_fitness:
            self._best_fitness = float(best.fitness)
            self._best_genome = best

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int | np.ndarray:
        if self._best_genome is None:
            genome = next(iter(self._population.population.values()))
        else:
            genome = self._best_genome
        output = self._activate(genome, np.asarray(obs, dtype=np.float32))
        if self._discrete:
            return int(np.argmax(output))
        return np.array(
            [np.tanh(v) * self._action_scale for v in output], dtype=np.float32
        ).reshape(-1)

    def save(self, model_path: str | Path) -> None:
        path = Path(model_path).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._best_genome is None:
            data: dict = {"best_fitness": self._best_fitness, "nodes": [], "connections": []}
        else:
            g = self._best_genome
            nodes = [
                {"key": k, "bias": v.bias, "response": v.response, "activation": v.activation}
                for k, v in g.nodes.items()
            ]
            connections = [
                {"key": list(k), "weight": v.weight, "enabled": v.enabled, "innovation": v.innovation}
                for k, v in g.connections.items()
            ]
            data = {"best_fitness": self._best_fitness, "nodes": nodes, "connections": connections}
        path.write_text(json.dumps(data))

    def load(self, model_path: str | Path, env=None) -> None:
        path = Path(model_path).with_suffix(".json")
        data = json.loads(path.read_text())
        self._best_fitness = float(data["best_fitness"])
        if not data["nodes"]:
            return

        genome = neat.DefaultGenome(0)
        genome.configure_new(self._config.genome_config)
        genome.nodes.clear()
        genome.connections.clear()

        for n in data["nodes"]:
            node = neat.genes.DefaultNodeGene(n["key"])
            node.bias = float(n["bias"])
            node.response = float(n["response"])
            node.activation = n["activation"]
            node.aggregation = "sum"
            genome.nodes[n["key"]] = node

        for c in data["connections"]:
            key = tuple(c["key"])
            conn = neat.genes.DefaultConnectionGene(key, innovation=int(c["innovation"]))
            conn.weight = float(c["weight"])
            conn.enabled = bool(c["enabled"])
            genome.connections[key] = conn

        genome.fitness = self._best_fitness
        self._best_genome = genome

    @property
    def policy_table(self) -> np.ndarray | None:
        if not self._discrete:
            return None
        from src.envs.state_utils import augment_state
        if self._best_genome is None:
            return None
        pos_bins = np.linspace(-1.2, 0.6, 24)
        vel_bins = np.linspace(-0.07, 0.07, 24)
        table = np.zeros((24, 24), dtype=np.int32)
        for i, pos in enumerate(pos_bins):
            for j, vel in enumerate(vel_bins):
                obs = np.array([pos, vel], dtype=np.float32)
                if self._augment:
                    obs = augment_state(obs)
                output = self._activate(self._best_genome, obs)
                table[i, j] = int(np.argmax(output))
        return table
