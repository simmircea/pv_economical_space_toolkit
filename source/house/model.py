"""
House and appliance models for energy consumption simulation.

This module provides classes for modeling household energy consumption,
including individual appliances and their consumption patterns.
"""

from dataclasses import dataclass
from datetime import datetime
from source.house.constants import APPLIANCES


@dataclass
class TimeRange:
    """
    A time range with start and end times.
    The start and end times are inclusive
    and are defined in seconds since the start of the simulation.
    """

    start_time: datetime
    end_time: datetime

    def __post_init__(self) -> None:
        """
        Post-initialization method to ensure the start time
        is before the end time.
        The start and end times are inclusive
        and are defined in seconds since the start of the simulation.
        """
        if self.start_time > self.end_time:
            raise ValueError("Start time must be before end time")

    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp is within this time range."""
        return self.start_time <= timestamp <= self.end_time

    def overlaps_with(self, other: 'TimeRange') -> bool:
        """Check if this time range overlaps with another."""
        return not (self.end_time < other.start_time
                    or other.end_time < self.start_time)

    def duration_hours(self) -> float:
        """Get duration in hours."""
        return (self.end_time - self.start_time).total_seconds() / 3600


@dataclass
class ConsumptionData:
    """
    Container for consumption data.
    The consumption data is stored in 10-minute intervals.
    The data is stored in kW.

    Attributes:
        usage_10min: Dictionary of 10-minute interval consumption data in kW
        usage_hourly: Optional dictionary of hourly consumption data in kW
    """

    usage_10min: dict[datetime, float]
    usage_hourly: dict[datetime, float] | None = None


@dataclass
class HouseDbRecord:
    """Data transfer object for house database records.

    The house_id represents the unique identifier
    for the house in the database.
    The zip_code represents the postal code of the house location.
    The location represents the name of the house location.
    The weather_station_id represents the ID of the nearest weather station.
    The start_epoch_time represents
    the start time of the consumption recordings for the house in the database.
    The end_epoch_time represents the end time of the consumption recordings
    for the house in the database.
    """
    house_id: int
    zip_code: str
    location: str
    weather_station_id: int
    start_epoch_time: int
    end_epoch_time: int


@dataclass
class WeatherStationDbRecord:
    """Data transfer object for weather station database records."""

    weather_station_id: int
    location: str
    latitude: float
    longitude: float


@dataclass
class ApplianceDbRecord:
    """Data transfer object for appliance database records.

    The appliance_id represents the unique identifier
    for the appliance in the database.
    The house_id represents the unique identifier
    for the house in the database.
    The name represents the name of the appliance.
    """
    appliance_id: int
    house_id: int
    name: str


class House:
    """
     This class represents a house in the simulation,
    managing its appliances and consumption data.

    Attributes:
        house_id: Unique identifier for the house
        zip_code: Postal code of the house location
        location: Name of the house location
        weather_station_id: ID of the nearest weather station
        time_range: Time range of consumption recordings for the house
        appliances: List of appliances in the house
        consumption: Consumption data for the house
    """

    def __init__(self, house_id: int,
                 zip_code: str | None = None,
                 location: str | None = None,
                 weather_station_id: int | None = None) -> None:
        """
        Initialize a new House instance.

        Args:
            house_id: Unique identifier for the house
            zip_code: Postal code of the house location
            location: Name of the house location
            weather_station_id: ID of the nearest weather station
        """
        self.house_id: int = house_id
        self.zip_code: str | None = zip_code
        self.location: str | None = location
        self.weather_station_id: int | None = weather_station_id
        self.time_range: TimeRange

        self.appliances: list[Appliance] = []
        self.consumption: ConsumptionData


class Appliance:
    """
    This class represents an individual appliance and its energy
    consumption patterns at different time intervals.

    Attributes:
        ID: Unique identifier for the appliance
        house: Reference to the parent House instance
        name: Name of the appliance
        type: Type of appliance (from APPLIANCES enum)
        consumption: Consumption data for the appliance
    """

    def __init__(self, ID: int,
                 house: House,
                 name: str,
                 type: APPLIANCES) -> None:
        """
        Initialize a new Appliance instance.

        Args:
            ID: Unique identifier for the appliance
            house: Reference to the parent House instance
            name: Name of the appliance
            type: Type of appliance from APPLIANCES enum
        """
        self.appliance_id: int = ID
        self.house: House = house
        self.name: str = name
        self.type: APPLIANCES = type
        self.consumption: ConsumptionData
