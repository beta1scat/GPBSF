"""Common interface used by the training engine."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ClassificationAdapter(ABC):
    """Wrap one upstream model without changing its source tree."""

    name: str

    @property
    @abstractmethod
    def model(self):
        raise NotImplementedError

    @abstractmethod
    def logits(self, points_bnc):
        """Return unnormalized Bx3 logits from a float32 BxNx3 tensor."""
        raise NotImplementedError

    @abstractmethod
    def loss(self, logits, labels):
        raise NotImplementedError

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
