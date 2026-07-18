"""Module 2: Student Profile Generator.

Assigns each of `config.students` synthetic students to a profile archetype
(weighted by `config.profile_distribution`) and produces their persistent
`Student` parameters via `ProfileFactory`.
"""

from __future__ import annotations

import numpy as np

from dataset_generator.config import GeneratorConfig
from dataset_generator.generators.profiles import ProfileFactory
from dataset_generator.models.student import Student
from dataset_generator.utils.rng import RNGStreams


def generate_students(config: GeneratorConfig, rng_streams: RNGStreams) -> list[Student]:
    """Generate `config.students` unique `Student`s.

    Profile membership for all students is drawn once from
    `rng_streams.student_rng`, weighted by `config.profile_distribution`
    (sorted keys, so the draw is deterministic for a given seed). Each
    student's numeric parameters are then sampled by its `BaseProfile`
    subclass using a per-student RNG (see `BaseProfile.generate_student`),
    so results are reproducible per-seed and independent of iteration order.
    """

    profile_keys = sorted(config.profile_distribution)
    weights = np.array([config.profile_distribution[key] for key in profile_keys])
    weights = weights / weights.sum()

    assigned_profiles = rng_streams.student_rng.choice(
        profile_keys, size=config.students, p=weights
    )

    students: list[Student] = []
    seen_ids: set[str] = set()
    for student_index, profile_key in enumerate(assigned_profiles):
        profile = ProfileFactory.create(str(profile_key))
        student = profile.generate_student(student_index, config)
        if student.student_id in seen_ids:
            raise RuntimeError(f"duplicate student_id generated: {student.student_id}")
        seen_ids.add(student.student_id)
        students.append(student)

    return students
