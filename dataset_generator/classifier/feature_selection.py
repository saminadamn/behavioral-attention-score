"""Module 8, Step 2: Feature Selection.

`FeatureSelector` returns a list of column *names* — it never touches an
actual `DatasetRecord` or DataFrame, so "never mutate DatasetRecord" holds
trivially. It's built entirely on top of Module 7's `FeatureRegistry`
(category + dtype per feature already exists there) rather than re-deriving
either from scratch, which is exactly the "never duplicate" instruction
applied to feature-selection logic.
"""

from __future__ import annotations

from dataset_generator.models.dataset import FeatureCategory
from dataset_generator.pipeline.feature_registry import FeatureRegistry


class FeatureSelector:
    """Selects `DatasetRecord` column names for model input."""

    def __init__(self, registry: FeatureRegistry | None = None) -> None:
        self._registry = registry or FeatureRegistry()

    def manual(self, feature_names: list[str]) -> list[str]:
        """Use exactly `feature_names`, validated against the registry, no filtering applied."""

        for name in feature_names:
            self._registry.get(name)  # raises KeyError if unknown
        return list(feature_names)

    def select(
        self,
        *,
        categories: list[FeatureCategory] | None = None,
        exclude_identifiers: bool = True,
        exclude_targets: bool = True,
        exclude_text: bool = True,
        numeric_only: bool = False,
        categorical_only: bool = False,
        whitelist: list[str] | None = None,
        blacklist: list[str] | None = None,
    ) -> list[str]:
        """Select feature names via category/dtype filters, then whitelist/blacklist refinement.

        Filter order: start from `categories` (or every category if `None`)
        -> drop identifiers/targets if requested -> drop free-text fields
        if requested -> restrict to numeric/categorical dtype if requested
        -> intersect with `whitelist` if given -> remove `blacklist`
        entries. Each stage is independent, so combinations (e.g. "numeric
        Behaviour features minus one column") compose predictably.

        `exclude_text` defaults to `True`: free-text fields (`prompt_text`,
        `response_text`, `prompt_keywords`, `prompt_learning_objective`,
        `response_hesitation_markers` — dtype `"text"`, distinct from
        bounded `"str"` categoricals like `prompt_subject`) need NLP
        vectorization to be useful model input, which is out of this
        module's scope. One-hot-encoding them directly (treating each
        near-unique string as its own category) is not just useless — it's
        a genuine performance trap: on a few thousand rows it produces a
        near-one-column-per-row encoding, which is what made an earlier,
        unfiltered version of this method hang. Use `.manual()` to opt into
        text columns explicitly if a future module adds real text
        vectorization.
        """

        candidate_categories = categories if categories is not None else self._registry.categories()
        selected = [
            d for c in candidate_categories for d in self._registry.by_category(c)
        ]

        if exclude_identifiers:
            selected = [d for d in selected if d.category != FeatureCategory.IDENTIFIER]
        if exclude_targets:
            selected = [d for d in selected if d.category != FeatureCategory.TARGET]
        if exclude_text:
            selected = [d for d in selected if d.dtype != "text"]

        if numeric_only:
            selected = [d for d in selected if d.dtype in ("int", "float")]
        if categorical_only:
            selected = [d for d in selected if d.dtype == "str"]

        names = [d.name for d in selected]

        if whitelist is not None:
            whitelist_set = set(whitelist)
            names = [n for n in names if n in whitelist_set]

        if blacklist is not None:
            blacklist_set = set(blacklist)
            names = [n for n in names if n not in blacklist_set]

        return names
