from source.pv.statistics import PVStatisticsLoader
from source.utils import TimeSpaceHandler
from source.pv.sizing import NEEGSizingStrategy
from source.pv.statistics import CandidateSpaceConfig
from source.house.creation import HouseBuilder

if __name__ == "__main__":

    # python pv_sizing_neeg.py

    time_space_handler = TimeSpaceHandler(location="Bucharest",
                                          start_date="01/02/1998",
                                          end_date="01/02/1999")
    house_id = 2000901

    # You can modify the max_surface_area_m2 parameter as needed
    # Default is 100 m² but you can change it based on your constraints
    candidate_space_config = CandidateSpaceConfig(house_id=house_id)

    statistics = PVStatisticsLoader().load_statistics_from_csv(house_id)
    house = HouseBuilder().build_house_by_id(house_id)

    NEEGSizingStrategy(time_space_handler=time_space_handler,
                       candidate_space_config=candidate_space_config,
                       statistics=statistics).run()
