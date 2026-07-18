"""Module 7, Step 2: Dataset Builder.

Converts already-simulated `SessionRecord`s into flat `DatasetRecord`s. This
class performs pure assembly — it reads fields off `Student`/`Prompt`/
`Response`/`BehaviourRecord`/`SessionRecord` and copies them into one row;
it never samples, computes a new statistic, or calls back into any
generator. That's what "must never regenerate behaviour" means concretely:
if a value isn't already sitting on one of those objects, it doesn't belong
in `DatasetRecord` (see `models/dataset.py`'s docstring for the fields that
were deliberately left out for exactly this reason).
"""

from __future__ import annotations

from dataset_generator.models.dataset import DatasetRecord
from dataset_generator.models.session import InteractionRecord, SessionRecord
from dataset_generator.models.student import Student


class DatasetBuilder:
    """Builds `DatasetRecord`s from completed `SessionRecord`s + their `Student`s.

    `students` is dependency-injected (a lookup, not constructed here) since
    `SessionRecord` only carries `student_id`/`student_profile` — the static
    per-student fields (`baseline_latency`, etc.) live on the `Student`
    object the session simulator was given, which the builder needs handed
    to it explicitly.
    """

    def __init__(self, students: list[Student]) -> None:
        self._students_by_id = {student.student_id: student for student in students}

    def build(self, sessions: list[SessionRecord]) -> list[DatasetRecord]:
        """Flatten every interaction across `sessions` into one `DatasetRecord` each."""

        records: list[DatasetRecord] = []
        for session in sessions:
            student = self._students_by_id.get(session.student_id)
            if student is None:
                raise KeyError(
                    f"session {session.session_id!r} references student "
                    f"{session.student_id!r}, which was not provided to DatasetBuilder"
                )
            for interaction in session.interactions:
                records.append(self._build_record(session, student, interaction))
        return records

    def _build_record(
        self, session: SessionRecord, student: Student, interaction: InteractionRecord
    ) -> DatasetRecord:
        prompt = interaction.prompt
        response = interaction.response
        behaviour = interaction.behaviour
        summary = session.summary

        return DatasetRecord(
            session_id=session.session_id,
            student_id=student.student_id,
            interaction_number=interaction.interaction_number,
            prompt_id=prompt.prompt_id,
            response_id=response.response_id,
            attention_state=behaviour.attention_state.value,
            intervention_applied=behaviour.intervention_applied,
            session_progress=behaviour.metadata.session_progress,
            student_profile=student.profile_name,
            student_profile_description=student.description,
            student_baseline_latency=student.baseline_latency,
            student_latency_variance=student.latency_variance,
            student_engagement_tendency=student.engagement_tendency,
            student_fatigue_rate=student.fatigue_rate,
            student_intervention_sensitivity=student.intervention_sensitivity,
            prompt_subject=prompt.subject,
            prompt_topic=prompt.topic,
            prompt_difficulty=prompt.difficulty.value,
            prompt_cognitive_level=prompt.cognitive_level.value,
            prompt_text=prompt.prompt_text,
            prompt_expected_answer_type=prompt.expected_answer_type,
            prompt_estimated_response_length=prompt.estimated_response_length,
            prompt_keywords="|".join(prompt.keywords),
            prompt_learning_objective=prompt.learning_objective,
            prompt_reading_time_seconds=prompt.metadata.estimated_reading_time_seconds,
            prompt_token_count=prompt.metadata.token_count,
            prompt_concept_count=prompt.metadata.concept_count,
            prompt_difficulty_score=prompt.metadata.difficulty_score,
            prompt_cognitive_complexity_score=prompt.metadata.cognitive_complexity_score,
            prompt_readability_grade=prompt.metadata.readability_grade,
            response_text=response.response_text,
            response_correctness_score=response.correctness_score,
            response_length=response.response_length,
            response_semantic_similarity=response.semantic_similarity,
            response_lexical_diversity=response.lexical_diversity,
            response_sentiment=response.sentiment,
            response_engagement_proxy=response.engagement_proxy,
            response_confidence=response.confidence,
            response_hesitation_markers="|".join(response.hesitation_markers),
            response_repetition_ratio=response.features.repetition_ratio,
            response_coherence_score=response.features.coherence_score,
            response_topic_shift=response.features.topic_shift,
            response_strategy_used=response.metadata.strategy_used,
            behaviour_response_latency=behaviour.response_latency,
            behaviour_interaction_duration=behaviour.interaction_duration,
            behaviour_hesitation_duration=behaviour.hesitation_duration,
            behaviour_rolling_latency=behaviour.rolling_latency,
            behaviour_rolling_engagement=behaviour.rolling_engagement,
            behaviour_fatigue_level=behaviour.fatigue_level,
            behaviour_normalized_latency=behaviour.features.normalized_latency,
            behaviour_transition_occurred=behaviour.features.transition_occurred,
            session_total_interactions=summary.total_interactions,
            session_dominant_attention_state=summary.dominant_attention_state,
            session_intervention_count=summary.intervention_count,
            session_final_fatigue=summary.final_fatigue,
            session_average_engagement=summary.average_engagement,
            session_average_correctness=summary.average_correctness,
            session_average_latency=summary.average_latency,
        )
