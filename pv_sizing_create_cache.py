
from source.pv.statistics import CandidateSpaceConfig, PVStatisticsExporter
from source.utils import TimeSpaceHandler

if __name__ == "__main__":
    # python pv_sizing_create_cache.py

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

    max_n_panels = 60
    candidate_space_config = CandidateSpaceConfig(
        house_ids=eligible_house_ids,
        n_panels_bounds=(1, max_n_panels),
        panel_type_bounds=(0, 2),
        mount_type_bounds=(0, 2),
        max_surface_area_m2=100)

    exporter = PVStatisticsExporter(
        candidate_space_config=candidate_space_config,
        time_space_handler=time_space_handler)

    exporter.export_all_statistics()
