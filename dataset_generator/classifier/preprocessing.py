"""Module 8, Step 3: Preprocessing Pipeline.

Wraps a single `sklearn.compose.ColumnTransformer` rather than hand-rolling
encoding/scaling/imputation — those are exactly the well-tested primitives
scikit-learn already provides, and reimplementing them would be its own
form of duplicated logic. `Preprocessor`'s job is deciding, from
`FeatureRegistry` dtypes (not ad hoc dtype-sniffing on whatever DataFrame
happens to be passed in), which columns get which treatment, and enforcing
one deterministic column order at both fit and transform time.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from dataset_generator.pipeline.feature_registry import FeatureRegistry


def _cast_to_float(x: pd.DataFrame) -> pd.DataFrame:
    """Boolean -> float cast for `FunctionTransformer` (module-level so joblib can pickle it —
    a `lambda` or nested/closure function here would make a fitted `Preprocessor` unsaveable).
    """

    return x.astype(float)


class Preprocessor:
    """Deterministic fit/transform pipeline over a fixed, ordered feature list.

    - Numeric features: median imputation + standard scaling.
    - Categorical (`str`) features: most-frequent imputation + one-hot
      encoding (`handle_unknown="ignore"`, so an unseen category at
      inference time produces an all-zero encoding rather than raising).
    - Boolean features: cast to `int` (no scaling — a boolean's two states
      are already on a meaningful, comparable scale).

    `feature_names` (the fixed input column order) is fixed at construction
    time and is exactly what should be persisted alongside a trained model
    (see `TrainingMetadata.feature_names`) — `transform` always reindexes
    its input to this order first, so it never depends on a caller's
    DataFrame having columns in any particular order.
    """

    def __init__(self, feature_names: list[str], registry: FeatureRegistry | None = None) -> None:
        self._registry = registry or FeatureRegistry()
        self.feature_names = list(feature_names)

        text_features = [f for f in self.feature_names if self._registry.get(f).dtype == "text"]
        if text_features:
            raise ValueError(
                f"Preprocessor does not support free-text features without NLP "
                f"vectorization (out of scope for this module): {text_features}. "
                "Remove them from feature_names, or use FeatureSelector.select() "
                "(exclude_text=True by default) instead of .manual()."
            )

        self._numeric_features = [
            f for f in self.feature_names if self._registry.get(f).dtype in ("int", "float")
        ]
        self._categorical_features = [
            f for f in self.feature_names if self._registry.get(f).dtype == "str"
        ]
        self._boolean_features = [
            f for f in self.feature_names if self._registry.get(f).dtype == "bool"
        ]
        self._column_transformer: ColumnTransformer | None = None
        self.feature_names_out_: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._column_transformer is not None

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Fit encoders/scalers on `df` (must contain every `feature_names` column)."""

        numeric_pipeline = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        boolean_pipeline = Pipeline(
            [
                # SimpleImputer rejects bool dtype directly; cast to float
                # first (True/False -> 1.0/0.0) so imputation is well-defined.
                ("cast", FunctionTransformer(_cast_to_float, feature_names_out="one-to-one")),
                ("impute", SimpleImputer(strategy="most_frequent")),
            ]
        )

        transformers = []
        if self._numeric_features:
            transformers.append(("numeric", numeric_pipeline, self._numeric_features))
        if self._categorical_features:
            transformers.append(("categorical", categorical_pipeline, self._categorical_features))
        if self._boolean_features:
            transformers.append(("boolean", boolean_pipeline, self._boolean_features))

        self._column_transformer = ColumnTransformer(transformers, remainder="drop")
        ordered = df.reindex(columns=self.feature_names)
        self._column_transformer.fit(ordered)
        self.feature_names_out_ = list(self._column_transformer.get_feature_names_out())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex `df` to `feature_names` and apply the fitted transform."""

        if self._column_transformer is None:
            raise RuntimeError("Preprocessor.transform called before fit()")
        ordered = df.reindex(columns=self.feature_names)
        transformed = self._column_transformer.transform(ordered)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return pd.DataFrame(transformed, columns=self.feature_names_out_, index=df.index)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: str | Path) -> Path:
        """Persist the fitted preprocessor (including `feature_names`) via joblib."""

        if self._column_transformer is None:
            raise RuntimeError("cannot save an unfitted Preprocessor")
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        return file_path

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        """Load a previously-saved `Preprocessor`."""

        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return loaded
