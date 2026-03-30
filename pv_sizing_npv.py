

from source.indicators.models import NPVConfig, npv, roi
from source.pv.sizing import EconomicalSizingStrategy
from source.pv.statistics import CandidateSpaceConfig, PVStatisticsLoader
from source.utils import TimeSpaceHandler


if __name__ == "__main__":
    # python pv_sizing_npv.py

    time_space_handler = TimeSpaceHandler(location="Grenoble",
                                          start_date="01/02/1998",
                                          end_date="01/02/1999")

    house_id = 2000901
    statistics = PVStatisticsLoader().load_statistics_from_csv(house_id)

    candidate_space_config = CandidateSpaceConfig(house_id=house_id)

    npv_config = NPVConfig(
        initial_cost_per_kW=1000,
        cost_per_kW=0.1,
        feed_in_price_per_kW=0.4)

    npv_sizing_strategy = EconomicalSizingStrategy(
        indicator_function=roi,
        time_space_handler=time_space_handler,
        candidate_space_config=candidate_space_config,
        npv_config=npv_config,
        statistics=statistics)

    npv_sizing_strategy.run()
