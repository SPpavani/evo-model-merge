"""
Shared architecture for all 'expert' models.

Critical design constraint (same as real-world model merging / Sakana AI's
Evolutionary Model Merge): every expert MUST share an identical architecture
and parameter shapes. You cannot merge weights of differently-shaped models —
this is why frameworks like mergekit only merge models from the same family
(e.g. all Mistral-7B variants, all Llama-3-8B variants).
"""
import torch
import torch.nn as nn


class ExpertNet(nn.Module):
    """A small feed-forward classifier used as a stand-in for a domain-expert LLM."""

    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_flat_params(self) -> torch.Tensor:
        """Flatten all parameters into a single vector — needed for SLERP/crossover math."""
        return torch.cat([p.data.flatten() for p in self.parameters()])

    def set_flat_params(self, flat: torch.Tensor) -> None:
        """Load a flat parameter vector back into the model's layers."""
        offset = 0
        for p in self.parameters():
            numel = p.data.numel()
            p.data.copy_(flat[offset:offset + numel].view_as(p.data))
            offset += numel
