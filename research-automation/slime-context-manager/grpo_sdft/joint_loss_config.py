"""Configuration object for the FASD-GRPO joint objective."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointLossConfig:
    lambda_sdft: float = 1.0
    lambda_bc: float = 0.1
    lambda_kl: float = 0.02

    def validate(self) -> None:
        for name, value in (
            ("lambda_sdft", self.lambda_sdft),
            ("lambda_bc", self.lambda_bc),
            ("lambda_kl", self.lambda_kl),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    def formula(self) -> str:
        return (
            "L_total = L_GRPO "
            f"+ {self.lambda_sdft:g} * L_SDFT "
            f"+ {self.lambda_bc:g} * L_replay "
            f"+ {self.lambda_kl:g} * L_KL"
        )
