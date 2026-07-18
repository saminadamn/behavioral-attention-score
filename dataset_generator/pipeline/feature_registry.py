"""Module 7, Step 3: Feature Registry.

A single, centralized description of every `DatasetRecord` field — its
category, dtype, and meaning — so downstream consumers (an attention
classifier choosing input features, a paper table describing the schema,
this project's own tests) have one place to ask "what features exist" and
"which ones are targets vs. inputs" instead of re-deriving it from
`DatasetRecord`'s source each time.
"""

from __future__ import annotations

from dataset_generator.models.dataset import FeatureCategory, FeatureDefinition

SCHEMA_VERSION = "1.0"

FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    # Identifiers
    FeatureDefinition("session_id", FeatureCategory.IDENTIFIER, "str", "Session identifier."),
    FeatureDefinition("student_id", FeatureCategory.IDENTIFIER, "str", "Student identifier."),
    FeatureDefinition("interaction_number", FeatureCategory.IDENTIFIER, "int", "Position within the session, from 1."),
    FeatureDefinition("prompt_id", FeatureCategory.IDENTIFIER, "str", "Prompt identifier."),
    FeatureDefinition("response_id", FeatureCategory.IDENTIFIER, "str", "Response identifier."),
    # Target
    FeatureDefinition("attention_state", FeatureCategory.TARGET, "str", "Ground-truth attention state for this interaction."),
    FeatureDefinition("response_correctness_score", FeatureCategory.TARGET, "float", "Continuous correctness estimate in [0,1]; a secondary regression target."),
    # Session (interaction-level context + session-level aggregates)
    FeatureDefinition("intervention_applied", FeatureCategory.SESSION, "bool", "Whether an intervention was applied at this interaction."),
    FeatureDefinition("session_progress", FeatureCategory.SESSION, "float", "Fraction of the session elapsed, in (0,1]."),
    FeatureDefinition("session_total_interactions", FeatureCategory.SESSION, "int", "Total interactions in this session."),
    FeatureDefinition("session_dominant_attention_state", FeatureCategory.SESSION, "str", "Most frequent attention state across the session."),
    FeatureDefinition("session_intervention_count", FeatureCategory.SESSION, "int", "Total interventions applied across the session."),
    FeatureDefinition("session_final_fatigue", FeatureCategory.SESSION, "float", "Fatigue level at the session's last interaction."),
    FeatureDefinition("session_average_engagement", FeatureCategory.SESSION, "float", "Mean engagement_proxy across the session."),
    FeatureDefinition("session_average_correctness", FeatureCategory.SESSION, "float", "Mean correctness_score across the session."),
    FeatureDefinition("session_average_latency", FeatureCategory.SESSION, "float", "Mean response_latency across the session."),
    # Student
    FeatureDefinition("student_profile", FeatureCategory.STUDENT, "str", "Student archetype name."),
    FeatureDefinition("student_profile_description", FeatureCategory.STUDENT, "str", "Human-readable archetype description."),
    FeatureDefinition("student_baseline_latency", FeatureCategory.STUDENT, "float", "Student's personal baseline response latency (s)."),
    FeatureDefinition("student_latency_variance", FeatureCategory.STUDENT, "float", "Student's personal latency std (s)."),
    FeatureDefinition("student_engagement_tendency", FeatureCategory.STUDENT, "float", "Student's baseline engagement tendency in [0,1]."),
    FeatureDefinition("student_fatigue_rate", FeatureCategory.STUDENT, "float", "Student's per-session fatigue accumulation rate."),
    FeatureDefinition("student_intervention_sensitivity", FeatureCategory.STUDENT, "float", "Student's responsiveness to interventions."),
    # Prompt
    FeatureDefinition("prompt_subject", FeatureCategory.PROMPT, "str", "Curriculum subject key."),
    FeatureDefinition("prompt_topic", FeatureCategory.PROMPT, "str", "Curriculum topic key."),
    FeatureDefinition("prompt_difficulty", FeatureCategory.PROMPT, "str", "Easy / Medium / Hard."),
    FeatureDefinition("prompt_cognitive_level", FeatureCategory.PROMPT, "str", "Bloom's taxonomy level."),
    FeatureDefinition("prompt_text", FeatureCategory.PROMPT, "str", "The prompt's rendered text."),
    FeatureDefinition("prompt_expected_answer_type", FeatureCategory.PROMPT, "str", "Expected answer shape (e.g. short_answer)."),
    FeatureDefinition("prompt_estimated_response_length", FeatureCategory.PROMPT, "int", "Target response length in tokens."),
    FeatureDefinition("prompt_keywords", FeatureCategory.PROMPT, "str", "Prompt's domain keywords, '|'-joined."),
    FeatureDefinition("prompt_learning_objective", FeatureCategory.PROMPT, "str", "The topic's learning objective sentence."),
    FeatureDefinition("prompt_reading_time_seconds", FeatureCategory.PROMPT, "float", "Estimated silent-reading time."),
    FeatureDefinition("prompt_token_count", FeatureCategory.PROMPT, "int", "Prompt text token count."),
    FeatureDefinition("prompt_concept_count", FeatureCategory.PROMPT, "int", "Number of keywords sampled for this prompt."),
    FeatureDefinition("prompt_difficulty_score", FeatureCategory.PROMPT, "float", "Computed difficulty score in [0,1]."),
    FeatureDefinition("prompt_cognitive_complexity_score", FeatureCategory.PROMPT, "float", "Bloom-level position in [0,1]."),
    FeatureDefinition("prompt_readability_grade", FeatureCategory.PROMPT, "float", "Flesch-Kincaid grade level."),
    # Response
    FeatureDefinition("response_text", FeatureCategory.RESPONSE, "str", "The generated student response text."),
    FeatureDefinition("response_length", FeatureCategory.RESPONSE, "int", "Response token count."),
    FeatureDefinition("response_semantic_similarity", FeatureCategory.RESPONSE, "float", "Concept coverage of [keyword, topic] in the response."),
    FeatureDefinition("response_lexical_diversity", FeatureCategory.RESPONSE, "float", "Type-token ratio of the response."),
    FeatureDefinition("response_sentiment", FeatureCategory.RESPONSE, "float", "Fixed-lexicon sentiment score in [-1,1]."),
    FeatureDefinition("response_engagement_proxy", FeatureCategory.RESPONSE, "float", "Composite engagement estimate."),
    FeatureDefinition("response_confidence", FeatureCategory.RESPONSE, "float", "How assertive the response sounds."),
    FeatureDefinition("response_hesitation_markers", FeatureCategory.RESPONSE, "str", "Detected hesitation phrases, '|'-joined."),
    FeatureDefinition("response_repetition_ratio", FeatureCategory.RESPONSE, "float", "Fraction of repeated bigrams."),
    FeatureDefinition("response_coherence_score", FeatureCategory.RESPONSE, "float", "Lexical-diversity/repetition-based coherence proxy."),
    FeatureDefinition("response_topic_shift", FeatureCategory.RESPONSE, "float", "Dissimilarity from the previous response."),
    FeatureDefinition("response_strategy_used", FeatureCategory.RESPONSE, "str", "Which `ResponseStrategy` class generated this response."),
    # Behaviour
    FeatureDefinition("behaviour_response_latency", FeatureCategory.BEHAVIOUR, "float", "Sampled, student-personalized response latency (s)."),
    FeatureDefinition("behaviour_interaction_duration", FeatureCategory.BEHAVIOUR, "float", "Total sampled interaction duration (s)."),
    FeatureDefinition("behaviour_hesitation_duration", FeatureCategory.BEHAVIOUR, "float", "Sampled hesitation-event-equivalent duration."),
    FeatureDefinition("behaviour_rolling_latency", FeatureCategory.BEHAVIOUR, "float", "EMA-smoothed latency trend."),
    FeatureDefinition("behaviour_rolling_engagement", FeatureCategory.BEHAVIOUR, "float", "EMA-smoothed engagement trend."),
    FeatureDefinition("behaviour_fatigue_level", FeatureCategory.BEHAVIOUR, "float", "Accumulated session fatigue in [0,1]."),
    FeatureDefinition("behaviour_normalized_latency", FeatureCategory.BEHAVIOUR, "float", "Z-score of latency vs. the student's own baseline."),
    FeatureDefinition("behaviour_transition_occurred", FeatureCategory.BEHAVIOUR, "bool", "Whether the attention state changed from the previous interaction."),
)


class FeatureRegistry:
    """Lookup interface over `FEATURE_DEFINITIONS`."""

    def __init__(self, definitions: tuple[FeatureDefinition, ...] = FEATURE_DEFINITIONS) -> None:
        self._by_name = {d.name: d for d in definitions}
        self._definitions = definitions

    @property
    def schema_version(self) -> str:
        return SCHEMA_VERSION

    def get(self, name: str) -> FeatureDefinition:
        """Look up one feature's definition by name."""

        if name not in self._by_name:
            raise KeyError(f"unknown feature {name!r}")
        return self._by_name[name]

    def all_features(self) -> tuple[FeatureDefinition, ...]:
        return self._definitions

    def by_category(self, category: FeatureCategory) -> list[FeatureDefinition]:
        """All feature definitions in `category`."""

        return [d for d in self._definitions if d.category == category]

    def categories(self) -> list[FeatureCategory]:
        """Every category with at least one feature, in definition order."""

        seen: list[FeatureCategory] = []
        for definition in self._definitions:
            if definition.category not in seen:
                seen.append(definition.category)
        return seen

    def feature_counts(self) -> dict[str, int]:
        """Number of features per category."""

        return {category.value: len(self.by_category(category)) for category in self.categories()}

    def numeric_features(self) -> list[str]:
        """Feature names whose dtype is int/float — the columns statistics/correlation apply to."""

        return [d.name for d in self._definitions if d.dtype in ("int", "float")]
