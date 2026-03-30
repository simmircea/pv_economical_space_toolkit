import csv
from dataclasses import dataclass
import dataclasses
from datetime import datetime
from typing import Any

from batem.core import solar


from source.house.creation import HouseBuilder
from source.house.model import House
from source.house.services import ConsumptionAggregator, ConsumptionTrimmer
from source.indicators.models import neeg, self_consumption, self_sufficiency
from source.pv.creation import (
    PVPlantBuilder, PVSizingFilePathBuilder, WeatherDataBuilder)
from source.pv.model import (
    PANEL_EFFICIENCY_MAP, PANEL_SURFACE_AREA_M2, PANEL_TYPE_TO_POWER_kW_MAP,
    PVConfig, PanelTypes, get_mount_type_from_index, get_panel_type_from_index)
from source.utils import TimeSpaceHandler


@dataclass
class CandidateSpaceConfig:
    """
    Configuration for the candidate space.
    Each candidate is a tuple of (n_panels, panel_type, mount_type).
    The bounds are the minimum and maximum values for each dimension.
    The max_surface_area_m2 is the maximum surface area in square meters.
    n_panels_bounds[1] is derived from max_surface_area_m2 so the search
    space matches the other project (max_panels = surface / panel_area).
    """
    house_ids: list[int]
    house_id: int | None = None
    n_panels_bounds: tuple[int, int] = (1, 200)
    panel_type_bounds: tuple[int, int] = (0, 2)
    mount_type_bounds: tuple[int, int] = (0, 2)
    max_surface_area_m2: float = 100.0

    def __post_init__(self) -> None:
        """Cap panel upper bound by max surface area."""
        max_n = int(self.max_surface_area_m2 / PANEL_SURFACE_AREA_M2)
        if self.n_panels_bounds[1] > max_n:
            object.__setattr__(
                self,
                "n_panels_bounds",
                (self.n_panels_bounds[0], max_n),
            )


@dataclass
class PVStatistics:
    """
    Statistics of a PV plant configuration.

    Attributes:
        panel_type: Type of the panel
        mount_type: Type of the mount
        n_panels: Number of panels
        peak_power_kW: Peak power of the PV plant
        surface_area_m2: Surface area of the PV plant
        efficiency: Efficiency of the PV plant
        annual_production_kwh: Annual production of the PV plant
        house_id: ID of the house
        self_consumption: Self-consumption of the PV plant over a year
        for the given house
    """
    panel_type: PanelTypes
    mount_type: solar.MOUNT_TYPES
    n_panels: int
    peak_power_kW: float
    surface_area_m2: float
    efficiency: float
    annual_production_kwh: float
    house_id: int
    self_consumption: float
    neeg_value: float
    total_production_kwh: float
    total_consumption_kwh: float
    self_sufficiency: float

    def to_list(self) -> list[Any]:
        """
        Convert the statistics to a list.
        """
        result = []
        for field in dataclasses.fields(PVStatistics):
            value = getattr(self, field.name)
            # Handle enum types by extracting their value
            if isinstance(value, (PanelTypes, solar.MOUNT_TYPES)):
                result.append(value.value)
            elif isinstance(value, (int, float)):
                result.append(value)
            else:
                result.append(value)
        return result


class PVStatisticsExporter:
    """
    Generate a CSV file with the statistics of PV plants
    for multiple configurations.
    """

    def __init__(self,
                 candidate_space_config: CandidateSpaceConfig,
                 time_space_handler: TimeSpaceHandler):

        self._candidate_space_config = candidate_space_config
        self._panel_types = self._get_panel_types()

        self._mount_types = self._get_mount_types()

        self._max_n_panels = self._candidate_space_config.n_panels_bounds[1]
        self._max_surface_area_m2 = \
            self._candidate_space_config.max_surface_area_m2

        self._fp_builder = PVSizingFilePathBuilder()

        # Prepare the consumption data associated with the house
        self._houses: dict[int, House] = {}

        for house_id in self._candidate_space_config.house_ids:
            house = HouseBuilder().build_house_by_id(house_id)
            self._houses[house_id] = house
            trimmer = ConsumptionTrimmer(house)
            trimmer.trim_consumption_house(time_space_handler)
            aggregator = ConsumptionAggregator(house)
            house.consumption.usage_hourly = \
                aggregator.get_total_consumption_hourly()

        # Prepare the weather data
        self._weather_data = WeatherDataBuilder().build(time_space_handler)
        self._solar_model = solar.SolarModel(self._weather_data)
        self._pv_plant_builder = PVPlantBuilder()

        # Set tracker
        self._n_records_to_export = len(self._panel_types) * \
            len(self._mount_types) * self._max_n_panels * \
            len(self._candidate_space_config.house_ids)
        self._n_records_exported = 0

    def _get_panel_types(self) -> list[PanelTypes]:
        return [get_panel_type_from_index(i) for i in range(
            self._candidate_space_config.panel_type_bounds[0],
            self._candidate_space_config.panel_type_bounds[1] + 1)]

    def _get_mount_types(self) -> list[solar.MOUNT_TYPES]:
        return [get_mount_type_from_index(i) for i in range(
            self._candidate_space_config.mount_type_bounds[0],
            self._candidate_space_config.mount_type_bounds[1] + 1)]

    def export_all_statistics(self):
        """
        Export the statistics of all PV plant configurations to a CSV file.
        Production and metrics are computed per (panel_type, n_panels,
        mount_type) by running the PV simulation for each configuration.
        """
        print(f"Exporting {self._n_records_to_export} records")

        csv_path = self._fp_builder.get_pv_statistics_csv_path()
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [field.name for field in dataclasses.fields(PVStatistics)])

            for house_id in self._candidate_space_config.house_ids:
                for panel_type in self._panel_types:
                    for mount_type in self._mount_types:
                        for n_panels in range(1, self._max_n_panels + 1):

                            # Check if the surface area is too large
                            surface_area_m2 = n_panels * PANEL_SURFACE_AREA_M2
                            if surface_area_m2 > self._max_surface_area_m2:
                                continue

                            # Get the statistics (production from full sim)
                            statistics = self.get_statistics_one_configuration(
                                panel_type=panel_type,
                                mount_type=mount_type,
                                n_panels=n_panels,
                                house=self._houses[house_id])
                            writer.writerow(statistics.to_list())

                            # Update the number of records exported
                            self._n_records_exported += 1
                            pct = (self._n_records_exported /
                                   self._n_records_to_export * 100)
                            print(
                                f"Exported {self._n_records_exported}/"
                                f"{self._n_records_to_export} records "
                                f"({pct:.2f}%)")

    def get_statistics_one_configuration(self,
                                         panel_type: PanelTypes,
                                         mount_type: solar.MOUNT_TYPES,
                                         n_panels: int,
                                         house: House):
        """
        Get the statistics of a single PV plant configuration.

        Production and metrics are computed by running the PV simulation
        for this exact (panel_type, n_panels, mount_type).
        """
        power_per_panel = PANEL_TYPE_TO_POWER_kW_MAP[panel_type]
        peak_power_kW = power_per_panel * n_panels
        surface_area_m2 = n_panels * PANEL_SURFACE_AREA_M2
        efficiency = PANEL_EFFICIENCY_MAP[panel_type]

        pv_config = PVConfig(
            panel_type=panel_type,
            number_of_panels=n_panels,
            mount_type=mount_type,
            pv_efficiency=efficiency)
        pv_plant = self._pv_plant_builder.build_from_solar_model(
            weather_data=self._weather_data,
            solar_model=self._solar_model,
            config=pv_config)
        usage_hourly = pv_plant.production.usage_hourly
        annual_production_kwh = sum(usage_hourly.values())

        consumption = house.consumption.usage_hourly
        if not consumption:
            raise ValueError("Consumption is not set")

        sc_value = self_consumption(consumption, usage_hourly)

        ss_value = self_sufficiency(consumption, usage_hourly)

        neeg_value = neeg(consumption, usage_hourly)
        total_production_kwh = sum(usage_hourly.values())

        total_consumption_kwh = sum(consumption.values())

        return PVStatistics(
            panel_type=panel_type,
            mount_type=mount_type,
            n_panels=n_panels,
            peak_power_kW=peak_power_kW,
            surface_area_m2=surface_area_m2,
            efficiency=efficiency,
            annual_production_kwh=annual_production_kwh,
            self_consumption=sc_value,
            house_id=house.house_id,
            neeg_value=neeg_value,
            total_production_kwh=total_production_kwh,
            total_consumption_kwh=total_consumption_kwh,
            self_sufficiency=ss_value
        )


class PVStatisticsLoader:
    """
    Load the statistics from a CSV file.
    """

    def __init__(self,):
        self._fp_builder = PVSizingFilePathBuilder()

    def load_statistics_from_csv(self) -> dict[
            tuple[int, PanelTypes, solar.MOUNT_TYPES, int], PVStatistics]:
        """
        Load the statistics from a CSV file.
        """

        csv_path = self._fp_builder.get_pv_statistics_csv_path()
        statistics_dict = {}
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                # Skip the header row
                if reader.line_num == 1:
                    continue
                statistics = PVStatistics(
                    panel_type=PanelTypes(row[0]),
                    mount_type=solar.MOUNT_TYPES(int(row[1])),
                    n_panels=int(row[2]),
                    peak_power_kW=float(row[3]),
                    surface_area_m2=float(row[4]),
                    efficiency=float(row[5]),
                    annual_production_kwh=float(row[6]),
                    house_id=int(row[7]),
                    self_consumption=float(row[8]),
                    neeg_value=float(row[9]),
                    total_production_kwh=float(row[10]),
                    total_consumption_kwh=float(row[11]),
                    self_sufficiency=float(row[12]))
                house_id = statistics.house_id
                panel_type = statistics.panel_type
                mount_type = statistics.mount_type
                n_panels = statistics.n_panels
                statistics_dict[(house_id, panel_type,
                                 mount_type, n_panels)] = statistics
        print(f"Loaded {len(statistics_dict)} statistics")
        return statistics_dict
