"""Module 8, Step 5: Classifier Models.

One thin wrapper per algorithm, all behind the same `ClassifierModel`
interface, registered with `ClassifierModelFactory` — the same
decorator-registry pattern used throughout this project
(`ProfileFactory`, `ResponseStrategyFactory`) rather than an
`if model_name == ...` chain. XGBoost/LightGBM are optional dependencies;
their wrappers are only registered if the package is importable, and
requesting one that isn't installed raises a clear error rather than an
opaque `ImportError` from deep inside the factory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class ClassifierModel(ABC):
    """Common interface every wrapped classifier implements."""

    name: ClassVar[str] = ""

    def __init__(self, random_state: int = 42, **kwargs: object) -> None:
        self.random_state = random_state
        self._estimator = self._build_estimator(random_state, **kwargs)

    @abstractmethod
    def _build_estimator(self, random_state: int, **kwargs: object): ...

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClassifierModel":
        self._estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._estimator.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        return self._estimator.classes_

    @property
    def underlying_estimator(self):
        """The wrapped scikit-learn/XGBoost/LightGBM estimator (for e.g. `feature_importances_`)."""

        return self._estimator

    def save(self, path: str | Path) -> Path:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        return file_path

    @classmethod
    def load(cls, path: str | Path) -> "ClassifierModel":
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, ClassifierModel):
            raise TypeError(f"{path} does not contain a ClassifierModel")
        return loaded


class ClassifierModelFactory:
    """Registry mapping a model name to its `ClassifierModel` implementation."""

    _registry: ClassVar[dict[str, type[ClassifierModel]]] = {}

    @classmethod
    def register(cls, model_cls: type[ClassifierModel]) -> type[ClassifierModel]:
        cls._registry[model_cls.name] = model_cls
        return model_cls

    @classmethod
    def create(cls, name: str, random_state: int = 42, **kwargs: object) -> ClassifierModel:
        if name not in cls._registry:
            raise KeyError(f"no classifier registered for {name!r}; known: {sorted(cls._registry)}")
        return cls._registry[name](random_state=random_state, **kwargs)

    @classmethod
    def available_models(cls) -> list[str]:
        return sorted(cls._registry)


@ClassifierModelFactory.register
class LogisticRegressionModel(ClassifierModel):
    name = "logistic_regression"

    def _build_estimator(self, random_state: int, **kwargs: object):
        return LogisticRegression(max_iter=1000, random_state=random_state, **kwargs)


@ClassifierModelFactory.register
class RandomForestModel(ClassifierModel):
    name = "random_forest"

    def _build_estimator(self, random_state: int, **kwargs: object):
        return RandomForestClassifier(n_estimators=200, random_state=random_state, **kwargs)


@ClassifierModelFactory.register
class GradientBoostingModel(ClassifierModel):
    name = "gradient_boosting"

    def _build_estimator(self, random_state: int, **kwargs: object):
        return GradientBoostingClassifier(random_state=random_state, **kwargs)


try:
    import xgboost as _xgboost

    @ClassifierModelFactory.register
    class XGBoostModel(ClassifierModel):
        name = "xgboost"

        def _build_estimator(self, random_state: int, **kwargs: object):
            return _xgboost.XGBClassifier(
                random_state=random_state, eval_metric="mlogloss", **kwargs
            )

except ImportError:
    pass


try:
    import lightgbm as _lightgbm

    @ClassifierModelFactory.register
    class LightGBMModel(ClassifierModel):
        name = "lightgbm"

        def _build_estimator(self, random_state: int, **kwargs: object):
            return _lightgbm.LGBMClassifier(random_state=random_state, verbose=-1, **kwargs)

except ImportError:
    pass
