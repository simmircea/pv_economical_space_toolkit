"""
Usage: python get_indicators_for_triplet.py
"""

from source.indicators.evaluation import (
    calculate_indicators_for_cost_triplet, export_triplet_indicators_to_csv)
from source.house.creation import HouseBuilder
from source.indicators.models import NPVConfig, neeg, npv, roi
from source.pv.creation import (
    PVPlantBuilder, PVSizingFilePathBuilder, WeatherDataBuilder)
from source.pv.sizing import EconomicalSizingStrategy, NEEGSizingStrategy
from source.pv.statistics import CandidateSpaceConfig
from source.utils import TimeSpaceHandler
import os
if __name__ == "__main__":

    eligible_house_ids = [2000900,
                          2000901,
                          2000903,
                          2000904,
                          2000905,
                          2000906]

    indicator = "roi"
    ts_handler = TimeSpaceHandler(location="Grenoble",
                                  start_date="22/01/1998",
                                  end_date="01/02/1999")

    weather_data = WeatherDataBuilder().build(ts_handler)

    # Initialize the indicators list

    indicators_list = []

    for id in eligible_house_ids:
        for min_objective in [npv, roi, neeg]:
            house = HouseBuilder().build_house_by_id(id)
            config = CandidateSpaceConfig(house_ids=[id])

            npv_config = NPVConfig(
                initial_cost_per_kW=779,
                cost_per_kW=0.18,
                feed_in_price_per_kW=0.091)

            if min_objective is neeg:
                sizing_strategy = NEEGSizingStrategy(
                    time_space_handler=ts_handler,
                    candidate_space_config=config,
                    statistics=None,
                    house=house)
            else:
                sizing_strategy = EconomicalSizingStrategy(
                    indicator_function=min_objective,
                    time_space_handler=ts_handler,
                    candidate_space_config=config,
                    npv_config=npv_config,
                    statistics=None,
                    house=house)

            optimal_pv_config, optimal_output = \
                sizing_strategy.run_without_statistics()

            plant = PVPlantBuilder().build(
                weather_data=weather_data,
                config=optimal_pv_config)

            output = calculate_indicators_for_cost_triplet(
                house=house,
                min_objective=min_objective.__name__,
                pv_plant=plant,
                initial_cost_per_kW=npv_config.initial_cost_per_kW,
                cost_per_kW=npv_config.cost_per_kW,
                feed_in_price_per_kW=npv_config.feed_in_price_per_kW,
                battery=None,
            )

            indicators_list.append(output)
            print(
                f"Indicators for house {id} and {min_objective.__name__} "
                f"calculated: {output}")

    builder = PVSizingFilePathBuilder()
    results_folder = builder.get_pv_sizing_folder()
    csv_path = os.path.join(results_folder, "indicators for triplet.csv")
    export_triplet_indicators_to_csv(indicators_list, csv_path)
