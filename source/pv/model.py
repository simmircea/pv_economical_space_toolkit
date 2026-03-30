

import dataclasses
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Any
from batem.core import weather
from batem.core import solar
from batem.core.solar import MOUNT_TYPES


from source.constants import DATE_FORMAT


@dataclass
class ProductionData:
    """
    Container for production data.
    The production data is stored in hourly intervals.
    The data is stored in kW.
    """
    usage_hourly: dict[datetime, float]


class PanelTypes(Enum):
    """
    Panel tyoess, with repsective power ratings in watts.
    """
    LOW_COST = "low_cost"
    STANDARD = "standard"
    HIGH_EFFICIENCY = "high_efficiency"


PANEL_TYPE_TO_POWER_kW_MAP = {
    PanelTypes.LOW_COST: 0.35,       # ~350 W/panou (≈21% pe 1.7 m²)
    PanelTypes.STANDARD: 0.375,      # ~375 W/panou (≈22% pe 1.7 m²)
    PanelTypes.HIGH_EFFICIENCY: 0.39  # ~390 W/panou (≈22.9% pe 1.7 m²)
}

PANEL_EFFICIENCY_MAP = {
    PanelTypes.LOW_COST: 0.15,
    PanelTypes.STANDARD: 0.20,
    PanelTypes.HIGH_EFFICIENCY: 0.25
}

PANEL_TYPE_TO_COST_FACTOR = {
    PanelTypes.LOW_COST: 0.8,        # 80% of the standard cost
    PanelTypes.STANDARD: 1.0,        # 100% of the standard cost
    PanelTypes.HIGH_EFFICIENCY: 1.2  # 120% of the standard cost
}

# Typical panel dimensions
# Panel height in meters
PANEL_HEIGHT_M = 1.7
# Panel width in meters
PANEL_WIDTH_M = 1.0
# Panel surface area in square meters
PANEL_SURFACE_AREA_M2 = PANEL_HEIGHT_M * PANEL_WIDTH_M


def get_panel_type_from_index(index: int) -> PanelTypes:
    """
    Get the optimal panel type from an index.
    """
    # Ensure panel_index is within valid range
    index = max(0, min(index, 2))
    return list(PanelTypes)[index]


def get_mount_type_from_index(index: int) -> solar.MOUNT_TYPES:
    """
    Get the optimal mount type from an index.
    """
    # Ensure mount_type_index is within valid range
    index = max(0, min(index, 2))
    return solar.MOUNT_TYPES(index)


@dataclass
class PVConfig:
    """Configuration for PV plant creation.

    Attributes:
        peak_power_kW: Peak power of the PV plant in kW
        number_of_panels: Number of panels in the PV plant
        panel_type: Type of the panel
        panel_height_m: Height of the panels in meters
        panel_width_m: Width of the panels in meters
        pv_efficiency: Efficiency of the PV panels
        PV_inverter_efficiency: Efficiency of the PV inverters
        temperature_coefficient: Temperature coefficient of the PV panels
        exposure_deg: Exposure angle of the PV plant
        slope_deg: Slope angle of the PV plant
        distance_between_arrays_m: Distance between arrays in meters
        mount_type: Mount type of the PV plant (flat, back to )
    """

    number_of_panels: int
    panel_type: PanelTypes
    panel_height_m: float = PANEL_HEIGHT_M
    panel_width_m: float = PANEL_WIDTH_M
    pv_efficiency: float = 0.2
    PV_inverter_efficiency: float = 0.95
    temperature_coefficient: float = 0.0035
    exposure_deg: float = 0
    slope_deg: float = 160
    distance_between_arrays_m: float = PANEL_HEIGHT_M
    mount_type: MOUNT_TYPES = MOUNT_TYPES.FLAT

    def get_peak_power_kW(self) -> float:
        """
        Get the peak power of the PV plant.
        """
        power_per_panel = PANEL_TYPE_TO_POWER_kW_MAP[self.panel_type]
        return self.number_of_panels * power_per_panel

    def get_key(self) -> str:
        """
        Get the key of the PV config.
        """
        pv_power = self.get_peak_power_kW()
        number_of_panels = self.number_of_panels
        total_power = pv_power * number_of_panels
        pv_power_str = str(total_power).replace(".", "_")
        return f"{pv_power_str}_kw"

    def get_names(self) -> list[str]:
        """
        Get the names of the PV config,
        in the order of the fields.
        """
        return [field.name for field in dataclasses.fields(PVConfig)]

    def to_list(self) -> list[Any]:
        """
        Convert the PV config to a list.
        """
        return [getattr(self, field.name)
                if isinstance(getattr(self, field.name), (int, float))
                else getattr(self, field.name).value
                for field in dataclasses.fields(PVConfig)
                ]

    def to_json(self) -> dict[str, Any]:
        """
        Convert the PV config to a JSON dictionary.
        """
        return {
            field.name: getattr(self, field.name)
            if isinstance(getattr(self, field.name), (int, float))
            else getattr(self, field.name).value
            for field in dataclasses.fields(PVConfig)}


class PVPlant:
    def __init__(self,
                 weather_data: weather.SiteWeatherData,
                 solar_model: solar.SolarModel,
                 config: PVConfig):
        """
        The location is the location of the PV plant.
        The latitude and longitude are the latitude
        and longitude of the PV plant.
        The exposure is the angle between the South
        and the direction of the PV plant.
        I.e. 0° is South, -90° is West, 90° is East, 180° is North.
        The slope is the angle between the horizontal
        and the plane of the PV plant.
        I.e. 0° is horizontal, 90° is vertical.
        The PV efficiency is the efficiency of the PV panel.
        The panel width and height are the width and height of the PV panel.
        The number of panels per array is the number of panels in each array.
        The peak power is the peak power of the PV plant.
        The PV inverter efficiency is the efficiency of the PV inverter.
        The temperature coefficient is the temperature coefficient
        of the PV panel.
        The distance between arrays is the distance between
        the arrays of the PV plant.
        The mount type is the type of mount of the PV plant.
        The power production is the power production of the PV plant
        expressed in kWh.
        """

        self.weather_data = weather_data
        self.solar_model = solar_model
        self.config = config

        self.production: ProductionData

    def get_start_time(self) -> datetime:
        return list(self.production.usage_hourly.keys())[0]

    def get_end_time(self) -> datetime:
        return list(self.production.usage_hourly.keys())[-1]

    def get_plot_path_key(self) -> str:
        """
        Create a unique key for the plot path.
        """
        start_time_str = self.get_start_time().strftime(DATE_FORMAT)
        end_time_str = self.get_end_time().strftime(DATE_FORMAT)
        name = "pv_plant"
        name = f"{name}_{self.weather_data.location}"
        name = f"{name}_from_{start_time_str}_to_{end_time_str}"
        return name
