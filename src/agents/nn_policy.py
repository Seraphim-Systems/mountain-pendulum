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
        return np.concatenate([p.detach().cpu().numpy().ravel() for p in self.parameters()]).astype(np.float64)

    def set_weights(self, w: np.ndarray) -> None:
        with torch.no_grad():
            offset = 0
            for p in self.parameters():
                size = p.numel()
                p.copy_(torch.tensor(w[offset:offset + size], dtype=p.dtype).reshape(p.shape))
                offset += size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def batched_forward(self, obs_batch: torch.Tensor | np.ndarray, all_weights: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Forward pass for N individuals each with different weights."""
        device = next(self.parameters()).device
        is_numpy = isinstance(obs_batch, np.ndarray)
        x = torch.from_numpy(obs_batch.astype(np.float32)).to(device) if is_numpy else obs_batch.float().to(device)
        w = torch.from_numpy(all_weights.astype(np.float32)).to(device) if isinstance(all_weights, np.ndarray) else all_weights.float().to(device)
        
        N = len(x)
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
                    
        return x.cpu().numpy() if is_numpy else x


def build_policy(env, hidden_sizes: list[int] = [64, 64]) -> PolicyNetwork:
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        n_out = env.action_space.n
    else:
        n_out = env.action_space.shape[0]
    return PolicyNetwork(obs_dim, n_out, hidden_sizes)
