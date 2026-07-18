"""Independent, reproducible random-number-generator streams.

A single shared `random.seed(seed)` means any module that draws a random
number shifts the sequence every other module sees afterward — a
non-deterministic ordering bug waiting to happen as the pipeline grows
(Steps 2-6 each need randomness). Instead, `build_rng_streams` derives one
independent `numpy.random.Generator` per concern from a single master seed,
so e.g. changing how prompts are sampled can never change which attention
states get sampled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_STREAM_NAMES: tuple[str, ...] = ("student", "session", "prompt", "response", "noise")


@dataclass(frozen=True)
class RNGStreams:
    """One independent `numpy.random.Generator` per generation concern."""

    student_rng: np.random.Generator
    session_rng: np.random.Generator
    prompt_rng: np.random.Generator
    response_rng: np.random.Generator
    noise_rng: np.random.Generator


def student_local_rng(master_seed: int, student_index: int) -> np.random.Generator:
    """An RNG scoped to exactly one student, independent of generation order.

    Derived from `(master_seed, student_index)` rather than by advancing a
    shared stream, so student #42's sampled parameters are identical whether
    the run generates 100 students or 10,000, and regardless of what order
    students are generated in.
    """

    return np.random.default_rng(np.random.SeedSequence([master_seed, student_index]))


def student_local_seed(master_seed: int, student_index: int) -> int:
    """The integer seed backing `student_local_rng(master_seed, student_index)`.

    Recorded on `Student.profile_seed` so a single student's generation is
    independently reproducible and citable in the dataset's metadata.
    """

    seed_sequence = np.random.SeedSequence([master_seed, student_index])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def build_rng_streams(seed: int) -> RNGStreams:
    """Derive the five named RNG streams deterministically from `seed`.

    Uses `numpy.random.SeedSequence.spawn`, which guarantees the resulting
    streams are statistically independent of each other while remaining
    fully reproducible: the same `seed` always yields the same five
    sequences, in isolation from one another.
    """

    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(len(_STREAM_NAMES))
    streams = {
        name: np.random.default_rng(child)
        for name, child in zip(_STREAM_NAMES, child_sequences)
    }
    return RNGStreams(
        student_rng=streams["student"],
        session_rng=streams["session"],
        prompt_rng=streams["prompt"],
        response_rng=streams["response"],
        noise_rng=streams["noise"],
    )
