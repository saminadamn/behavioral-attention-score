"""Prioritized Experience Replay (Schaul, Quan, Antonoglou & Silver, 2016) —
a sum-tree buffer that samples transitions with probability proportional to
their TD-error magnitude, with importance-sampling correction so the
resulting gradient stays an (approximately) unbiased estimate.
"""

from __future__ import annotations

import numpy as np

from dataset_generator.rl_experimental.environment import Transition


class _SumTree:
    """Binary array-backed sum tree: leaves hold priorities, internal nodes
    hold subtree sums, giving O(log n) update and O(log n) prefix-sum
    sampling instead of the O(n) rescan a flat priority array would need.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data: list[Transition | None] = [None] * capacity
        self._write = 0
        self.n_entries = 0

    def add(self, priority: float, data: Transition) -> None:
        leaf_index = self._write + self.capacity - 1
        self.data[self._write] = data
        self.update(leaf_index, priority)
        self._write = (self._write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, leaf_index: int, priority: float) -> None:
        change = priority - self.tree[leaf_index]
        self.tree[leaf_index] = priority
        idx = leaf_index
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, cumulative: float) -> tuple[int, float, Transition]:
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if cumulative <= self.tree[left]:
                idx = left
            else:
                cumulative -= self.tree[left]
                idx = right
        data_index = idx - (self.capacity - 1)
        transition = self.data[data_index]
        assert transition is not None
        return idx, self.tree[idx], transition

    @property
    def total_priority(self) -> float:
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        if self.n_entries == 0:
            return 1.0
        return float(self.tree[self.capacity - 1 : self.capacity - 1 + self.n_entries].max())


class PrioritizedReplayBuffer:
    """`alpha` controls how strongly priority affects sampling probability
    (0 = uniform, matching plain experience replay; 1 = fully greedy on
    TD error). `beta` corrects the resulting sampling bias via
    importance-sampling weights, and is annealed from `beta_start` toward
    1.0 over training — early on, some bias is accepted in exchange for
    prioritizing learning on the transitions that surprise the network
    most; later, correction matters more as the policy stabilizes.
    """

    def __init__(
        self,
        capacity: int,
        seed: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 2000,
        priority_epsilon: float = 1e-3,
    ) -> None:
        self._tree = _SumTree(capacity)
        self._rng = np.random.default_rng(seed)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.priority_epsilon = priority_epsilon
        self._sample_step = 0

    def __len__(self) -> int:
        return self._tree.n_entries

    def push(self, transition: Transition) -> None:
        # New transitions get the current max priority, so every transition
        # is sampled at least once before its real TD error is known.
        self._tree.add(self._tree.max_priority, transition)

    def _beta(self) -> float:
        fraction = min(1.0, self._sample_step / self.beta_anneal_steps)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def sample(self, batch_size: int) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        """Returns `(transitions, tree_indices, importance_sampling_weights)`.
        `tree_indices` must be passed back to `update_priorities` after
        training on this batch.
        """

        n = len(self)
        if batch_size > n:
            raise ValueError(f"Requested batch of {batch_size} but buffer holds only {n}.")

        total = self._tree.total_priority
        segment = total / batch_size

        transitions: list[Transition] = []
        tree_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size)

        for i in range(batch_size):
            low, high = segment * i, segment * (i + 1)
            cumulative = self._rng.uniform(low, high)
            leaf_index, priority, transition = self._tree.get(cumulative)
            tree_indices[i] = leaf_index
            priorities[i] = priority
            transitions.append(transition)

        sampling_probabilities = priorities / total
        weights = (n * sampling_probabilities) ** (-self._beta())
        weights /= weights.max()  # normalize so the max weight is 1 (stabilizes the LR)

        self._sample_step += 1
        return transitions, tree_indices, weights

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + self.priority_epsilon) ** self.alpha
        for idx, priority in zip(tree_indices, priorities):
            self._tree.update(int(idx), float(priority))


class ReplayBuffer:
    """Uniform-sampling replay buffer, kept for comparison/ablation against
    `PrioritizedReplayBuffer` (`DQNConfig.use_prioritized_replay=False`).
    """

    def __init__(self, capacity: int, seed: int) -> None:
        from collections import deque

        self._buffer: "deque[Transition]" = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def __len__(self) -> int:
        return len(self._buffer)

    def sample(self, batch_size: int) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Requested batch of {batch_size} but buffer holds only {len(self._buffer)}."
            )
        indices = self._rng.choice(len(self._buffer), size=batch_size, replace=False)
        transitions = [self._buffer[int(i)] for i in indices]
        weights = np.ones(batch_size)
        return transitions, indices, weights

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        # Uniform replay has no notion of priority — no-op, kept so
        # DQNAgent can call it unconditionally regardless of buffer type.
        return
