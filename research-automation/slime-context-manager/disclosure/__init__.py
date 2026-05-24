"""Progressive evidence disclosure policy components."""

from .schemas import (
    ACTIONS,
    BudgetAllocationRecord,
    ContextSpan,
    DisclosureDecision,
    DisclosureRollout,
    DisclosureTarget,
    SlimeSFTRecord,
)
from .on_policy_buffer import (
    OnPolicyReplayBuffer,
    ReplayRecord,
    build_replay_records_from_targets,
    mix_replay_with_onpolicy,
)
from .slime_opd_loss import (
    OPDLossConfig,
    OPDLossOutput,
    action_to_index,
    compute_full_opd_loss,
    compute_opd_loss,
    index_to_action,
)

__all__ = [
    # Schemas
    "ACTIONS",
    "BudgetAllocationRecord",
    "ContextSpan",
    "DisclosureDecision",
    "DisclosureRollout",
    "DisclosureTarget",
    "SlimeSFTRecord",
    # Replay Buffer
    "OnPolicyReplayBuffer",
    "ReplayRecord",
    "build_replay_records_from_targets",
    "mix_replay_with_onpolicy",
    # OPD Loss
    "OPDLossConfig",
    "OPDLossOutput",
    "action_to_index",
    "compute_full_opd_loss",
    "compute_opd_loss",
    "index_to_action",
]
