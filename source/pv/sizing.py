from batem.core import solar
from batem.core.weather import SiteWeatherData
from scipy.optimize import differential_evolution

from source.battery.model import Battery
from source.house.model import House
from source.house.services import ConsumptionTrimmer, ConsumptionAggregator
from source.indicators.models import NPVConfig, neeg, self_consumption
from source.pv.creation import PVPlantBuilder, WeatherDataBuilder
from source.pv.model import (
    PANEL_EFFICIENCY_MAP,
    PANEL_SURFACE_AREA_M2,
    PVConfig,
    PVPlant, PanelTypes, get_mount_type_from_index, get_panel_type_from_index)
from source.pv.statistics import CandidateSpaceConfig, PVStatistics
from source.utils import TimeSpaceHandler
from typing import Callable


def get_house_neeg(house: House, pv_plant: PVPlant,
                   battery: Battery | None = None) -> float:
    """
    Get the NEEG of a house with a PV plant.
    The result is in kWh.
    """
    pv_production = pv_plant.production.usage_hourly

    if house.consumption.usage_hourly is None:
        print("Warning: No hourly consumption data")
        return 0
    load_by_time = house.consumption.usage_hourly
    battery_power_by_time = (
        battery.get_battery_power_by_time() if battery else {})
    return neeg(load_by_time, pv_production, battery_power_by_time)


def optimize(bounds: list,
             diff_ev_args: tuple,
             objective_function: Callable,
             verbose: bool = False):
    """
    Optimize a function using the differential evolution algorithm.

    Args:
        bounds: List of bounds for the variables.
        diff_ev_args: Arguments for the objective function.
        objective_function: Objective function to optimize.
    Returns:
        tuple: Optimal configuration and objective function value.
    """

    result = differential_evolution(
        objective_function,
        args=diff_ev_args,
        bounds=bounds,
        workers=1,
        maxiter=10,
        popsize=20,
        disp=verbose,
        polish=True
    )

    optimal_panels = int(result.x[0])
    optimal_panel_index = int(round(result.x[1]))
    optimal_mount_type_index = int(round(result.x[2]))

    # Get the optimal panel type and mount type
    optimal_panel_type = get_panel_type_from_index(
        optimal_panel_index)
    optimal_mount_type = get_mount_type_from_index(
        optimal_mount_type_index)
    optimal_efficiency = PANEL_EFFICIENCY_MAP[optimal_panel_type]

    optimal_pv_config = PVConfig(
        number_of_panels=optimal_panels,
        panel_type=optimal_panel_type,
        mount_type=optimal_mount_type,
        pv_efficiency=optimal_efficiency)

    return optimal_pv_config, result.fun


class NEEGSizingStrategy:
    def __init__(self,
                 time_space_handler: TimeSpaceHandler,
                 candidate_space_config: CandidateSpaceConfig,
                 statistics: dict[
                     tuple[int, PanelTypes, solar.MOUNT_TYPES, int],
                     PVStatistics] | None,
                 verbose: bool = False,
                 house: House | None = None):
        self._ts_handler = time_space_handler
        self._candidate_space_config = candidate_space_config
        self._statistics = statistics
        self._verbose = verbose
        self._house = house
        # (panels, panel_type, mount_type)
        self.bounds = [self._candidate_space_config.n_panels_bounds,
                       self._candidate_space_config.panel_type_bounds,
                       self._candidate_space_config.mount_type_bounds]

        if self._house is not None:
            trimmer = ConsumptionTrimmer(self._house)
            trimmer.trim_consumption_house(self._ts_handler)
            aggregator = ConsumptionAggregator(self._house)
            self._house.consumption.usage_hourly = \
                aggregator.get_total_consumption_hourly()

        if self._statistics is None:
            self._weather_data = WeatherDataBuilder().build(self._ts_handler)
        else:
            self._weather_data = None

        self.diff_ev_args = (self._candidate_space_config.house_id,
                             self._statistics,
                             self._candidate_space_config.max_surface_area_m2)

    def run(self):
        optimal_pv_config, optimal_neeg = optimize(
            self.bounds,
            self.diff_ev_args,
            self.objective_function,
            self._verbose)

        if self._verbose:
            print(f"Optimal NEEG: {optimal_neeg:.3f}")
            print(f"Optimal PV config: {optimal_pv_config}")

        return optimal_pv_config, optimal_neeg

    @staticmethod
    def objective_function(x,
                           house_id: int,
                           statistics_dict: dict[
                               tuple[int, PanelTypes, solar.MOUNT_TYPES, int],
                               PVStatistics],
                           max_surface_area_m2: float):
        number_of_panels = int(x[0])
        panel_index = int(round(x[1]))
        mount_type_index = int(round(x[2]))

        # Ensure panel_index is within valid range
        panel_type = get_panel_type_from_index(panel_index)
        mount_type = get_mount_type_from_index(mount_type_index)

        # Calculate total surface area
        total_surface_area_m2 = number_of_panels * PANEL_SURFACE_AREA_M2

        # Apply surface area constraint
        if total_surface_area_m2 > max_surface_area_m2:
            return float('inf')  # Penalty for exceeding surface constraint

        statistics = statistics_dict[(
            house_id, panel_type, mount_type, number_of_panels)]

        neeg_value = statistics.neeg_value

        return neeg_value

    def run_without_statistics(self):
        if self._statistics is not None:
            raise ValueError(
                "run_without_statistics should be used "
                "only when statistics is None")

        if self._house is None:
            raise ValueError(
                "House instance with hourly consumption is required "
                "for run_without_statistics")

        if not self._house.consumption.usage_hourly:
            raise ValueError(
                "House consumption.usage_hourly must be set "
                "for run_without_statistics")

        consumption_by_time = self._house.consumption.usage_hourly

        diff_ev_args = (
            self._candidate_space_config.house_id,
            consumption_by_time,
            self._candidate_space_config.max_surface_area_m2
        )

        optimal_pv_config, optimal_value = optimize(
            self.bounds,
            diff_ev_args,
            self.objective_function_without_statistics,
            self._verbose)

        if self._verbose:
            print(f"Optimal NEEG: {optimal_value:.3f}")
            print(f"Optimal PV config: {optimal_pv_config}")

        return optimal_pv_config, optimal_value

    def objective_function_without_statistics(
            self, x,
            house_id: int,
            consumption_by_time: dict,
            max_surface_area_m2: float):

        number_of_panels = int(x[0])
        panel_index = int(round(x[1]))
        mount_type_index = int(round(x[2]))

        panel_type = get_panel_type_from_index(panel_index)
        mount_type = get_mount_type_from_index(mount_type_index)
        efficiency = PANEL_EFFICIENCY_MAP[panel_type]

        total_surface_area_m2 = number_of_panels * PANEL_SURFACE_AREA_M2

        if total_surface_area_m2 > max_surface_area_m2:
            return float('inf')  # Penalty for exceeding surface constraint

        if self._weather_data is None:
            raise ValueError("Weather data is not set")

        pv_config = PVConfig(
            number_of_panels=number_of_panels,
            panel_type=panel_type,
            mount_type=mount_type,
            pv_efficiency=efficiency)

        plant = PVPlantBuilder().build_from_solar_model(
            weather_data=self._weather_data,
            solar_model=solar.SolarModel(self._weather_data),
            config=pv_config)

        # total_production_kwh = sum(plant.production.usage_hourly.values())

        # sc_value = self_consumption(
        #    consumption_by_time,
        #    plant.production.usage_hourly)

        neeg_value = neeg(
            consumption_by_time,
            plant.production.usage_hourly,
            battery_power_by_time={})

        return neeg_value


class EconomicalSizingStrategy:
    def __init__(self,
                 time_space_handler: TimeSpaceHandler,
                 candidate_space_config: CandidateSpaceConfig,
                 npv_config: NPVConfig,
                 indicator_function: Callable,
                 statistics: dict[
                     tuple[int, PanelTypes, solar.MOUNT_TYPES, int],
                     PVStatistics] | None,
                 house: House | None = None,
                 verbose: bool = False):

        self._ts_handler = time_space_handler
        self._candidate_space_config = candidate_space_config
        self._npv_config = npv_config
        self._indicator_function = indicator_function
        self._statistics = statistics if statistics is not None else None
        self._house = house
        if house is not None:
            trimmer = ConsumptionTrimmer(house)
            trimmer.trim_consumption_house(self._ts_handler)
            aggregator = ConsumptionAggregator(house)
            house.consumption.usage_hourly = \
                aggregator.get_total_consumption_hourly()
        if statistics is None:
            self._weather_data = WeatherDataBuilder().build(self._ts_handler)
        else:
            self._weather_data = None

        self._verbose = verbose

        self.bounds = [self._candidate_space_config.n_panels_bounds,
                       self._candidate_space_config.panel_type_bounds,
                       self._candidate_space_config.mount_type_bounds]

        self.diff_ev_args = (
            self._indicator_function,
            self._candidate_space_config.house_id,
            self._npv_config,
            self._statistics,
            self._weather_data,
            self._candidate_space_config.max_surface_area_m2
        )

    def run(self) -> tuple[PVConfig, float]:

        optimal_pv_config, optimal_value = optimize(
            self.bounds,
            self.diff_ev_args,
            self.objective_function,
            self._verbose)

        if self._verbose:
            print(f"Optimal output: {optimal_value:.3f}")
            print(f"Optimal PV config: {optimal_pv_config}")

        return optimal_pv_config, optimal_value

    @staticmethod
    def objective_function(x,
                           indicator_function: Callable,
                           house_id: int,
                           npv_config: NPVConfig,
                           statistics_dict: dict[
                               tuple[int, PanelTypes, solar.MOUNT_TYPES, int],
                               PVStatistics],
                           weather_data: SiteWeatherData | None,
                           max_surface_area_m2: float,):
        """
        Objective function for the NPV sizing strategy.
        Args:
            x: A vector of the number of panels,
            panel index and mount type index.
        """
        number_of_panels = int(x[0])
        panel_index = int(round(x[1]))
        mount_type_index = int(round(x[2]))

        panel_type = get_panel_type_from_index(panel_index)
        mount_type = get_mount_type_from_index(mount_type_index)
        efficiency = PANEL_EFFICIENCY_MAP[panel_type]

        total_surface_area_m2 = number_of_panels * PANEL_SURFACE_AREA_M2

        if total_surface_area_m2 > max_surface_area_m2:
            return float('inf')  # Penalty for exceeding surface constraint

        statistics = statistics_dict[(
            house_id, panel_type, mount_type, number_of_panels)]

        pv_config = PVConfig(
            number_of_panels=number_of_panels,
            panel_type=panel_type,
            mount_type=mount_type,
            pv_efficiency=efficiency)

        output = indicator_function(
            config=npv_config,
            pv_config=pv_config,
            annual_production_kwh=statistics.annual_production_kwh,
            sc_value=statistics.self_consumption
        )
        npv_value = output[0]
        return -npv_value

    def run_without_statistics(self) -> tuple[PVConfig, float]:
        if self._statistics is not None:
            raise ValueError(
                "run_without_statistics should be used "
                "only when statistics is None")

        if self._house is None:
            raise ValueError(
                "House instance with hourly consumption is required "
                "for run_without_statistics")

        if not self._house.consumption.usage_hourly:
            raise ValueError(
                "House consumption.usage_hourly must be set "
                "for run_without_statistics")

        consumption_by_time = self._house.consumption.usage_hourly

        diff_ev_args = (
            self._indicator_function,
            self._candidate_space_config.house_id,
            self._npv_config,
            consumption_by_time,
            self._weather_data,
            self._candidate_space_config.max_surface_area_m2
        )

        optimal_pv_config, optimal_value = optimize(
            self.bounds,
            diff_ev_args,
            self.objective_function_without_statistics,
            self._verbose)

        if self._verbose:
            print(f"Optimal output: {optimal_value:.3f}")
            print(f"Optimal PV config: {optimal_pv_config}")

        return optimal_pv_config, optimal_value

    @staticmethod
    def objective_function_without_statistics(
            x,
            indicator_function: Callable,
            house_id: int,
            npv_config: NPVConfig,
            consumption_by_time,
            weather_data: SiteWeatherData | None,
            max_surface_area_m2: float,):
        """
        Objective function for the NPV sizing strategy without statistics.

        This version builds the PV plant on the fly from weather data
        and uses its production directly. The self-consumption factor is
        approximated by assuming that all production is self-consumed
        (i.e. the load profile matches the production profile).
        """
        number_of_panels = int(x[0])
        panel_index = int(round(x[1]))
        mount_type_index = int(round(x[2]))

        panel_type = get_panel_type_from_index(panel_index)
        mount_type = get_mount_type_from_index(mount_type_index)
        efficiency = PANEL_EFFICIENCY_MAP[panel_type]

        total_surface_area_m2 = number_of_panels * PANEL_SURFACE_AREA_M2

        if total_surface_area_m2 > max_surface_area_m2:
            return float('inf')  # Penalty for exceeding surface constraint

        pv_config = PVConfig(
            number_of_panels=number_of_panels,
            panel_type=panel_type,
            mount_type=mount_type,
            pv_efficiency=efficiency)

        if weather_data is None:
            raise ValueError("Weather data is not set")

        plant = PVPlantBuilder().build_from_solar_model(
            weather_data=weather_data,
            solar_model=solar.SolarModel(weather_data),
            config=pv_config)

        total_production_kwh = sum(plant.production.usage_hourly.values())

        sc_value = self_consumption(
            consumption_by_time,
            plant.production.usage_hourly)

        output = indicator_function(
            config=npv_config,
            pv_config=pv_config,
            annual_production_kwh=total_production_kwh,
            sc_value=sc_value)
        npv_value = output[0]
        return -npv_value
