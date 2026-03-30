from source.indicators.models import neeg, npv, roi
from source.pv.economical_space import (
    EconomicalSpace, EconomicalSpaceEvaluator)
from source.utils import TimeSpaceHandler


if __name__ == "__main__":
    # python pv_sizing_space_evaluator.py

    # house_ids = [2000901, 2000907]
    eligible_house_ids = [2000900,
                          2000901,
                          2000903,
                          2000904,
                          2000905,
                          2000906]
    time_space_handler = TimeSpaceHandler(location="Grenoble",
                                          start_date="22/01/1998",
                                          end_date="01/02/1999")

    for indicator_function in [neeg, npv, roi]:
        print(f"Evaluating {indicator_function.__name__}...")
        economical_space = EconomicalSpace(
            initial_cost_per_kW_bounds=(750, 1250),
            initial_cost_increment=10,
            cost_per_kW_bounds=(0.1, 0.4),
            cost_per_kW_increment=0.05,
            feed_in_price_per_kW_bounds=(0.1, 0.4),
            feed_in_price_per_kW_increment=0.05,
            house_ids=eligible_house_ids,
            indicator_function=indicator_function,
            n_iterations_to_save=2)

        evaluator = EconomicalSpaceEvaluator(
            time_space_handler=time_space_handler,
            economical_space=economical_space,
            verbose=False,
            n_workers=4)

        evaluator.evaluate()
