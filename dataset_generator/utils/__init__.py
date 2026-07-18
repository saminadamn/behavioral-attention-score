"""Cross-cutting utilities used by multiple generator modules."""

from dataset_generator.utils.rng import (
    RNGStreams,
    build_rng_streams,
    student_local_rng,
    student_local_seed,
)
from dataset_generator.utils.heuristic_nlp import (
    concept_coverage,
    find_hesitation_markers,
    hesitation_marker_count,
    repetition_ratio,
    simple_sentiment,
)
from dataset_generator.utils.text_metrics import (
    count_syllables,
    estimate_reading_time_seconds,
    flesch_kincaid_grade,
    flesch_reading_ease,
    sentence_count,
    token_jaccard_similarity,
    word_count,
    word_tokenize,
)

__all__ = [
    "RNGStreams",
    "build_rng_streams",
    "concept_coverage",
    "count_syllables",
    "estimate_reading_time_seconds",
    "find_hesitation_markers",
    "flesch_kincaid_grade",
    "flesch_reading_ease",
    "hesitation_marker_count",
    "repetition_ratio",
    "sentence_count",
    "simple_sentiment",
    "student_local_rng",
    "student_local_seed",
    "token_jaccard_similarity",
    "word_count",
    "word_tokenize",
]
