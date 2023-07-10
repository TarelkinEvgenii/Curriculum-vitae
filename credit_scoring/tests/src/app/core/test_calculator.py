import numpy as np
import pytest

from src.app.core.api import Features
from src.app.core.calculator import Calculator


class TestCalculator(object):
    """Тест калькулятора."""

    @pytest.mark.parametrize(
        'proba, inc_per_adult, home_feature, annuity_to_income_ratio, years_old, expected_value',
        [
            (0, 20000, 1, 0.1, 25, 20000 * 8 * 1.2 * 1.2 * 1.2),
            (0.1, 20000, 1, 0.1, 25, 20000 * 5 * 1.2 * 1.2 * 1.2),
            (0.2, 20000, 1, 0.1, 25, 20000 * 3 * 1.2 * 1.2 * 1.2),
            (1, 20000, 1, 0.1, 25, 20000 * 3 * 1.2 * 1.2 * 1.2),
            (1, np.nan, 1, 0.1, 25, 10000 * 1.2 * 1.2 * 1.2),
            (1, np.nan, 0, 0.1, 25, 10000 * 0.8 * 1.2 * 1.2),
            (1, np.nan, 0, 0.1, 24, 10000 * 0.8 * 1.2 * 0.8),
            (1, np.nan, 0, 0.1, 50, 10000 * 0.8 * 1.2 * 0.8),
            (1, np.nan, 0, 0.2, 50, 10000 * 0.8 * 0.8 * 0.8),
        ],
    )
    def test_calc_amount(
            self,
            proba,
            inc_per_adult,
            home_feature,
            annuity_to_income_ratio,
            years_old,
            expected_value,
    ):
        """
        Для тестирования калькулятора.

        :param proba: граничная проба
        :param inc_per_adult: доход на взрослого
        :param home_feature: наличие дома
        :param annuity_to_income_ratio: соотношение платежей к зарплате
        :param years_old: количество
        :param expected_value: сколько хочет денег
        """
        features = Features(
            INC_PER_ADULT=inc_per_adult,
            HOME_FEATURE=home_feature,
            ANNUITY_TO_INCOME_RATIO=annuity_to_income_ratio,
            YEARS_OLD=years_old,
        )
        calculator = Calculator()  # чтобы МР можно было делать
        assert calculator.calc_amount(
            proba,
            features,
        ) == expected_value
