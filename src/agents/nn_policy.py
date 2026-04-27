import numpy as np
import torch
import torch.nn as nn
import gymnasium


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_out: int, hidden_sizes: list[int] = [64, 64]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_out))
        self.net = nn.Sequential(*layers)
        self.train(False)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_weights(self) -> np.ndarray:
        return np.concatenate([p.detach().numpy().ravel() for p in self.parameters()]).astype(np.float64)

    def set_weights(self, w: np.ndarray) -> None:
        with torch.no_grad():
            offset = 0
            for p in self.parameters():
                size = p.numel()
                p.copy_(torch.tensor(w[offset:offset + size], dtype=p.dtype).reshape(p.shape))
                offset += size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def batched_forward(self, obs_batch: np.ndarray, all_weights: np.ndarray) -> np.ndarray:
        """Forward pass for N individuals each with different weights.

        obs_batch:   (N, obs_dim)  float32 numpy
        all_weights: (N, n_params) float64 numpy
        returns:     (N, n_out)    float32 numpy
        """
        N = len(obs_batch)
        x = torch.from_numpy(obs_batch.astype(np.float32))       # (N, obs_dim)
        w = torch.from_numpy(all_weights.astype(np.float32))      # (N, n_params)
        offset = 0
        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    out_dim, in_dim = layer.weight.shape
                    W = w[:, offset:offset + out_dim * in_dim].reshape(N, out_dim, in_dim)
                    offset += out_dim * in_dim
                    b = w[:, offset:offset + out_dim]              # (N, out_dim)
                    offset += out_dim
                    # (N, out_dim, in_dim) × (N, in_dim, 1) → (N, out_dim)
                    x = torch.bmm(W, x.unsqueeze(-1)).squeeze(-1) + b
                elif isinstance(layer, nn.ReLU):
                    x = torch.relu(x)
        return x.numpy()


def build_policy(env, hidden_sizes: list[int] = [64, 64]) -> PolicyNetwork:
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        n_out = env.action_space.n
    else:
        n_out = env.action_space.shape[0]
    return PolicyNetwork(obs_dim, n_out, hidden_sizes)
