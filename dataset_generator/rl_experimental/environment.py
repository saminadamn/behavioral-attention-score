"""Turns already-computed Dataset/BAS/Reward/Intervention artifacts into
offline `(state_sequence, action, reward, next_state_sequence, done)`
transitions.

This is deliberately NOT a live environment: there is no student to act on
between training updates. It replays the logged sequence of observations,
the rule engine's own chosen actions, and the reward the reward engine
already computed for each interaction — standard offline/batch RL, with
the logging policy being `InterventionPlanner` itself. See the package
docstring in `__init__.py` for what this can and cannot demonstrate.

States are **sequences**, not single snapshots: each transition's state is
a window of the last `sequence_length` interactions (left-padded with the
earliest available frame when a session hasn't produced that many yet),
so the LSTM encoder in `network.py` has actual temporal context to use
instead of one interaction in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dataset_generator.bas.models import BASArtifact
from dataset_generator.intervention.models import InterventionArtifact, InterventionObservation
from dataset_generator.intervention.observation import InterventionObservationExtractor
from dataset_generator.models.dataset import DatasetArtifact
from dataset_generator.reward.models import RewardArtifact

# Fixed action ordering — must match InterventionPolicyFactory's registration
# order in dataset_generator/intervention/policies.py. Encoded explicitly
# (not derived from the factory at import time) so the action space is
# stable even if new policies are registered later.
ACTION_NAMES: tuple[str, ...] = (
    "NoInterventionPolicy",
    "HintPolicy",
    "ConceptReviewPolicy",
    "DifficultyReductionPolicy",
    "MotivationalPromptPolicy",
    "BreakRecommendationPolicy",
    "EncouragementPolicy",
    "QuestionReframingPolicy",
)
ACTION_INDEX: dict[str, int] = {name: i for i, name in enumerate(ACTION_NAMES)}

STATE_DIM = 12


def observation_to_state(observation: InterventionObservation) -> np.ndarray:
    """A fixed-order, already-bounded feature vector — no fitting, no
    scaling parameters to leak between train/eval, since every source
    field is already normalized to a known range upstream. This is one
    frame of a sequence, not the whole state the agent sees.
    """

    return np.array(
        [
            observation.current_bas,
            observation.bas_trend if observation.bas_trend is not None else 0.0,
            observation.current_reward,
            observation.reward_trend if observation.reward_trend is not None else 0.0,
            observation.fatigue,
            observation.engagement,
            min(observation.latency_deviation, 3.0) / 3.0,
            observation.correctness,
            observation.confidence,
            observation.semantic_similarity,
            observation.prompt_difficulty_score,
            observation.session_progress,
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class Transition:
    state: np.ndarray  # (sequence_length, STATE_DIM)
    action: int
    reward: float
    next_state: np.ndarray  # (sequence_length, STATE_DIM)
    done: bool


def _windowed_sequence(frames: list[np.ndarray], end_index: int, sequence_length: int) -> np.ndarray:
    """The window of `sequence_length` frames ending at (and including)
    `frames[end_index]`, left-padded by repeating `frames[0]` when the
    session hasn't produced enough interactions yet.
    """

    start = end_index - sequence_length + 1
    if start >= 0:
        window = frames[start : end_index + 1]
    else:
        pad = [frames[0]] * (-start)
        window = pad + frames[: end_index + 1]
    return np.stack(window)


def build_transitions(
    intervention_artifact: InterventionArtifact,
    reward_artifact: RewardArtifact,
    observations: list[InterventionObservation],
    sequence_length: int,
) -> list[Transition]:
    """One transition per interaction, grouped by session so a window
    never crosses a session boundary and the last interaction of each
    session is marked `done=True`.
    """

    reward_by_key = {(r.student_id, r.session_id, r.interaction_number): r.reward for r in reward_artifact.records}
    action_by_key = {
        (d.student_id, d.session_id, d.interaction_number): d.chosen_policy
        for d in intervention_artifact.decisions
    }

    by_session: dict[str, list[InterventionObservation]] = {}
    for obs in observations:
        by_session.setdefault(obs.session_id, []).append(obs)

    transitions: list[Transition] = []
    for session_id, session_obs in by_session.items():
        ordered = sorted(session_obs, key=lambda o: o.interaction_number)
        frames = [observation_to_state(o) for o in ordered]

        for i, obs in enumerate(ordered):
            key = (obs.student_id, session_id, obs.interaction_number)
            action_name = action_by_key.get(key)
            if action_name is None or action_name not in ACTION_INDEX:
                continue
            reward = reward_by_key.get(key)
            if reward is None:
                continue

            is_last = i == len(ordered) - 1
            state_seq = _windowed_sequence(frames, i, sequence_length)
            next_seq = _windowed_sequence(frames, i if is_last else i + 1, sequence_length)

            transitions.append(
                Transition(
                    state=state_seq,
                    action=ACTION_INDEX[action_name],
                    reward=reward,
                    next_state=next_seq,
                    done=is_last,
                )
            )
    return transitions


def collect_transitions(
    dataset_artifact: DatasetArtifact,
    bas_artifact: BASArtifact,
    reward_artifact: RewardArtifact,
    intervention_artifact: InterventionArtifact,
    sequence_length: int,
) -> list[Transition]:
    """The one place every offline-RL trainer in this package (DQN, CQL, IQL,
    BCQ) turns four artifacts into transitions — extracted so each trainer
    doesn't repeat the observation-extraction + windowing call pair.
    """

    observations = InterventionObservationExtractor().extract_batch(
        dataset_artifact, bas_artifact, reward_artifact
    )
    return build_transitions(
        intervention_artifact, reward_artifact, observations, sequence_length=sequence_length
    )
