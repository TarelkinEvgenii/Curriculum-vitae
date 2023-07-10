import dataclasses
import pickle

import pandas as pd

from src.app.core.api import Features, ScoringDecision, ScoringResult
from src.app.core.calculator import Calculator


class AdvancedModel(object):
    """Класс для модели, которая выбирает одобренную сумму."""

    _threshold = 0.3

    def __init__(self, model_path: str):
        """Создает объект класса.

        :param model_path: Путь к pickle файлу с моделью.
        """
        with open(model_path, 'rb') as pickled_model:
            self._model = pickle.load(pickled_model)
        self._calculator = Calculator()

    def get_scoring_result(self, features: Features) -> ScoringResult:
        """Вычисляет одобренную сумму.

        :param features: Фичи используемые для предсказания
        :return: Решение относительно текущего дела
        """
        proba = self._predict_proba(features)

        decision = ScoringDecision.DECLINED
        amount = 0
        if proba < self._threshold:
            decision = ScoringDecision.ACCEPTED
            amount = self._calculator.calc_amount(
                proba,
                features,
            )

        return ScoringResult(
            decision=decision,
            amount=amount,
            threshold=self._threshold,
            proba=proba,
        )

    def _predict_proba(self, features: Features) -> float:
        """Определяет вероятность невозврата займа.

        :param features: Фичи используемые для предсказания
        :return: Вероятность дефолта
        """
        # важен порядок признаков для catboost
        features = pd.DataFrame([dataclasses.asdict(features)])
        # Возвращаем порядок в DataFrame
        features = features[list(Features.__annotations__)]
        return self._model.predict_proba(features)[:, 1][0]
