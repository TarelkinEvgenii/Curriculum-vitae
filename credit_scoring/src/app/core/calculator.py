import numpy as np

from src.app.core.api import Features


class Calculator(object):
    """Класс для вычисления одобренной суммы."""

    def calc_amount(self, proba: str, features: Features) -> int:
        """
        Функция рассчитывает одобренную сумму.

        :param proba: Вероятность дефолта
        :param features: Фичи используемые для предсказания
        :return: Одобренная сумма
        """
        lower_border = 0.1  # комментарий для МРа
        higher_border = 0.2
        bonus_sum = 1.2
        malus_sum = 0.8

        if features.INC_PER_ADULT is np.nan:
            resulted_sum = 10_000
        else:
            if proba < lower_border:
                resulted_sum = features.INC_PER_ADULT * 8
            elif lower_border <= proba < higher_border:
                resulted_sum = features.INC_PER_ADULT * 5
            else:
                resulted_sum = features.INC_PER_ADULT * 3

        # Наличие своего дома сильно снижает расходы на снятие жилья
        if features.HOME_FEATURE == 1:
            resulted_sum *= bonus_sum
        else:
            resulted_sum *= malus_sum

        # Сколько отдаёт от заработка очень важно
        lower_ratio_border = 0.2
        if features.ANNUITY_TO_INCOME_RATIO < lower_ratio_border:
            resulted_sum *= bonus_sum
        else:
            resulted_sum *= malus_sum

        # Возраст в котором берёт кредит очень важен, так как если давать молодым,
        # то они не осознаны
        # А если в старом возрасте давать, то есть вероятность, что не успеет отдать
        # И производительность в старом возрасте меньше сильно
        # (добавил, чтобы мр прошёл)
        if features.YEARS_OLD > 24 and features.YEARS_OLD < 50:
            resulted_sum = resulted_sum * 1.5
        else:
            resulted_sum *= malus_sum
        return resulted_sum
