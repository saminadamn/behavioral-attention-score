"""Three algorithms purpose-built for offline data, as distinct from the
Double DQN + PER + LSTM upgrades in the parent package (which make fitting
logged data faster and more stable, but do not address the extrapolation
problem itself — see `docs/EXPERIMENTAL_DQN.md`):

- **CQL** (`cql.py`) — a conservative penalty pushes Q down on
  out-of-distribution actions, up on the logged action.
- **IQL** (`iql.py`) — an expectile value function avoids ever
  bootstrapping off an action the critic itself selects.
- **Discrete BCQ** (`bcq.py`) — a behavior-cloning model defines a support
  set; Q is never maximized over, and the induced policy never selects,
  an action outside it.

All three remain research prototypes, same status as the rest of
`rl_experimental/`: not part of the default pipeline, not validated
against real outcomes.
"""

from dataset_generator.rl_experimental.offline.bcq import (
    BCQAgent,
    BCQConfig,
    BCQTrainer,
    BCQTrainingArtifact,
    default_bcq_config,
    masked_bellman_targets,
    support_mask,
)
from dataset_generator.rl_experimental.offline.cql import (
    CQLAgent,
    CQLConfig,
    CQLTrainer,
    CQLTrainingArtifact,
    default_cql_config,
)
from dataset_generator.rl_experimental.offline.iql import (
    IQLAgent,
    IQLConfig,
    IQLTrainer,
    IQLTrainingArtifact,
    default_iql_config,
    expectile_weight,
)

__all__ = [
    "BCQAgent",
    "BCQConfig",
    "BCQTrainer",
    "BCQTrainingArtifact",
    "CQLAgent",
    "CQLConfig",
    "CQLTrainer",
    "CQLTrainingArtifact",
    "IQLAgent",
    "IQLConfig",
    "IQLTrainer",
    "IQLTrainingArtifact",
    "default_bcq_config",
    "default_cql_config",
    "default_iql_config",
    "expectile_weight",
    "masked_bellman_targets",
    "support_mask",
]
