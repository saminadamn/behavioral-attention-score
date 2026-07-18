"""Subject/topic hierarchy for prompt generation (Module 3, Steps 2/7).

Adding a subject or topic is a data change here (or in a loaded config
file), never a code change in the generator — `PromptGenerator` only ever
reads `CurriculumConfig`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TopicDefinition(BaseModel):
    """One topic within a subject: its domain keyword bank and objective template."""

    model_config = ConfigDict(frozen=True)

    name: str
    keywords: list[str] = Field(min_length=3)
    learning_objective_template: str = "Understand the concept of {topic}."

    @model_validator(mode="after")
    def _check_name(self) -> "TopicDefinition":
        if not self.name.strip():
            raise ValueError("topic name must not be empty")
        return self


class SubjectDefinition(BaseModel):
    """One subject: its topics, keyed by topic key, plus an optional learning progression."""

    model_config = ConfigDict(frozen=True)

    name: str
    topics: dict[str, TopicDefinition]
    progression: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_topics(self) -> "SubjectDefinition":
        if not self.topics:
            raise ValueError(f"subject {self.name!r} must define at least one topic")
        if self.progression:
            unknown = set(self.progression) - set(self.topics)
            if unknown:
                raise ValueError(
                    f"subject {self.name!r} progression references unknown topics: {unknown}"
                )
        return self


class CurriculumConfig(BaseModel):
    """The full subject hierarchy, keyed by subject key."""

    model_config = ConfigDict(frozen=True)

    subjects: dict[str, SubjectDefinition]

    @model_validator(mode="after")
    def _check_subjects(self) -> "CurriculumConfig":
        if not self.subjects:
            raise ValueError("curriculum must define at least one subject")
        return self

    def ordered_topics(self, subject_key: str) -> list[TopicDefinition]:
        """Topics for `subject_key` in learning-progression order.

        Falls back to dict-insertion order if the subject defines no
        explicit `progression` (Module 3, Step 7).
        """

        subject = self.subjects[subject_key]
        if subject.progression:
            return [subject.topics[key] for key in subject.progression]
        return list(subject.topics.values())

    def topic_key_for(self, subject_key: str, topic_name: str) -> str:
        for key, topic in self.subjects[subject_key].topics.items():
            if topic.name == topic_name:
                return key
        raise KeyError(f"no topic named {topic_name!r} in subject {subject_key!r}")


def default_curriculum() -> CurriculumConfig:
    """The reference 7-subject curriculum used by `default_config()`."""

    return CurriculumConfig(
        subjects={
            "Mathematics": SubjectDefinition(
                name="Mathematics",
                topics={
                    "Arithmetic": TopicDefinition(
                        name="Arithmetic",
                        keywords=[
                            "addition", "subtraction", "multiplication", "division",
                            "estimation", "rounding", "decimals", "ratio",
                        ],
                        learning_objective_template="Perform and reason about {topic} operations.",
                    ),
                    "Algebra": TopicDefinition(
                        name="Algebra",
                        keywords=[
                            "variable", "equation", "expression", "coefficient",
                            "linear", "inequality", "function", "substitution",
                        ],
                        learning_objective_template="Solve problems involving {topic}.",
                    ),
                    "Geometry": TopicDefinition(
                        name="Geometry",
                        keywords=[
                            "angle", "triangle", "area", "perimeter",
                            "polygon", "circle", "volume", "symmetry",
                        ],
                        learning_objective_template="Reason about shapes and measurement in {topic}.",
                    ),
                    "Probability": TopicDefinition(
                        name="Probability",
                        keywords=[
                            "chance", "outcome", "event", "sample space",
                            "odds", "combination", "frequency", "distribution",
                        ],
                        learning_objective_template="Reason about uncertainty using {topic}.",
                    ),
                },
                progression=["Arithmetic", "Algebra", "Geometry", "Probability"],
            ),
            "Science": SubjectDefinition(
                name="Science",
                topics={
                    "Physics": TopicDefinition(
                        name="Physics",
                        keywords=[
                            "force", "energy", "motion", "gravity",
                            "velocity", "acceleration", "friction", "momentum",
                        ],
                        learning_objective_template="Explain physical phenomena related to {topic}.",
                    ),
                    "Chemistry": TopicDefinition(
                        name="Chemistry",
                        keywords=[
                            "element", "compound", "reaction", "molecule",
                            "acid", "solution", "catalyst", "bond",
                        ],
                        learning_objective_template="Explain matter and change through {topic}.",
                    ),
                    "Biology": TopicDefinition(
                        name="Biology",
                        keywords=[
                            "cell", "organism", "photosynthesis", "ecosystem",
                            "gene", "mutation", "adaptation", "respiration",
                        ],
                        learning_objective_template="Explain living systems through {topic}.",
                    ),
                },
                progression=["Physics", "Chemistry", "Biology"],
            ),
            "Reading": SubjectDefinition(
                name="Reading",
                topics={
                    "Comprehension": TopicDefinition(
                        name="Comprehension",
                        keywords=[
                            "main idea", "detail", "inference", "summary",
                            "context", "prediction", "evidence", "purpose",
                        ],
                        learning_objective_template="Extract and reason about meaning through {topic}.",
                    ),
                    "Literary_Analysis": TopicDefinition(
                        name="Literary Analysis",
                        keywords=[
                            "character", "plot", "setting", "conflict",
                            "symbolism", "theme", "imagery", "narrator",
                        ],
                        learning_objective_template="Analyze texts using {topic}.",
                    ),
                },
                progression=["Comprehension", "Literary_Analysis"],
            ),
            "Vocabulary": SubjectDefinition(
                name="Vocabulary",
                topics={
                    "Word_Meaning": TopicDefinition(
                        name="Word Meaning",
                        keywords=[
                            "synonym", "antonym", "definition", "context clue",
                            "root word", "prefix", "suffix", "connotation",
                        ],
                        learning_objective_template="Determine and use {topic} accurately.",
                    ),
                },
                progression=["Word_Meaning"],
            ),
            "General_Knowledge": SubjectDefinition(
                name="General Knowledge",
                topics={
                    "History": TopicDefinition(
                        name="History",
                        keywords=[
                            "event", "timeline", "cause", "effect",
                            "civilization", "revolution", "empire", "reform",
                        ],
                        learning_objective_template="Reason about past events through {topic}.",
                    ),
                    "Geography": TopicDefinition(
                        name="Geography",
                        keywords=[
                            "continent", "climate", "region", "landform",
                            "population", "border", "resource", "migration",
                        ],
                        learning_objective_template="Reason about places and environments through {topic}.",
                    ),
                },
                progression=["History", "Geography"],
            ),
            "Critical_Thinking": SubjectDefinition(
                name="Critical Thinking",
                topics={
                    "Logical_Reasoning": TopicDefinition(
                        name="Logical Reasoning",
                        keywords=[
                            "premise", "conclusion", "argument", "evidence",
                            "assumption", "fallacy", "inference", "bias",
                        ],
                        learning_objective_template="Reason rigorously using {topic}.",
                    ),
                },
                progression=["Logical_Reasoning"],
            ),
            "Problem_Solving": SubjectDefinition(
                name="Problem Solving",
                topics={
                    "Word_Problems": TopicDefinition(
                        name="Word Problems",
                        keywords=[
                            "strategy", "step", "solution", "estimate",
                            "plan", "variable", "check", "pattern",
                        ],
                        learning_objective_template="Apply a systematic strategy to {topic}.",
                    ),
                },
                progression=["Word_Problems"],
            ),
        }
    )
