"""
This module provides classes for creating House and Appliance instances
from various data sources (database, CSV files) and managing their
relationships.
"""

from dataclasses import dataclass
from datetime import datetime


import numpy


from source.house.model import (
    Appliance, ApplianceDbRecord, ConsumptionData, House,
    HouseDbRecord, TimeRange)
from source.house.services import (
    ApplianceInferrer, ConsumptionAggregatorAppliance,
    ConsumptionAggregator)
from source.utils import FilePathBuilder, epochtimems_to_datetime
from source.house.repositories import (
    ApplianceRepositoryCSV, ApplianceRepositorySQLite, HouseRepositorySQLite)


@dataclass
class BuilderConfig:
    """Configuration for house building operations.

    Attributes:
        db_path: Path to the database
        outlier_sensitivity: Sensitivity for outlier detection
        anomaly_threshold: Threshold for anomaly detection
        exclude_consumption: Whether to exclude consumption data;
        sometimes it's usefull just to know the time range of the house
    """

    db_path: str | None = None
    outlier_sensitivity: float = 10.0
    anomaly_threshold: float = 1500.0
    exclude_consumption: bool = False

    def get_db_path(self) -> str | None:
        """Get database path with fallback."""
        if self.db_path is None:
            return FilePathBuilder().get_irise_db_path()
        return self.db_path


class HouseBuilder:
    """
    Builder for creating House instances from various data sources.

    This class provides methods to create House instances from:
    - Database records
    - CSV files
    - Individual house IDs
    """

    def __init__(self) -> None:
        """Initialize the HouseBuilder."""
        pass

    def build_houses_from_db_records(self,
                                     db_path: str | None = None,
                                     exclude_consumption: bool = False
                                     ) -> list[House]:
        """
        Build all houses from the database.

        Args:
            exclude_consumption: Whether to exclude consumption data
                (default: False)

        Returns:
            list[House]: List of created House instances or
            None if no appliances found

        Example:
            >>> builder = HouseBuilder()
            >>> houses = builder.build_all_houses()
        """

        if db_path is None:
            db_path = FilePathBuilder().get_irise_db_path()

        records = HouseRepositorySQLite(db_path).get_all_house_records()

        houses: list[House] = []

        for record in records:
            house = self.build_house_from_db_record(
                record, db_path, exclude_consumption)
            if house is not None:
                houses.append(house)
        return houses

    def build_house_from_db_record(self,
                                   house_db_record: HouseDbRecord,
                                   db_path: str | None = None,
                                   exclude_consumption: bool = False
                                   ) -> House | None:
        """
        Build a house from database record.

        Args:
            house_data: Tuple containing house information:
                (ID, ZIPcode, Location, WeatherStationIDREF,
                 StartingEpochTime, EndingEpochTime)
            exclude_consumption: Whether to exclude consumption data
                (default: False)

        Returns:
            Optional[House]: Created House instance or None if no
                appliances found

        Example:
            >>> data = (1, "38000", "Grenoble", 1, 1234567890, 1234567899)
            >>> house = builder.build_house(data)
        """

        if db_path is None:
            db_path = FilePathBuilder().get_irise_db_path()

        # Create the house model and set the time range
        house = self._create_house_from_db_record(house_db_record)

        # Get the appliances first to check if the house has any appliances
        appliances = ApplianceBuilder().build_appliances_from_db(
            house, db_path)
        house.appliances = appliances

        if len(appliances) == 0:
            # If the house has no appliances, return None
            return None

        # If consumption is excluded, return the house
        # Sometimes it's usefull just to know the time range of the house
        if exclude_consumption:
            return house

        self._load_consumption(house)

        return house

    def _create_house_from_db_record(self, house_db_record: HouseDbRecord
                                     ) -> House:
        """
        Create a house from a database record.
        """
        # Create the house model
        house = House(house_id=house_db_record.house_id,
                      zip_code=house_db_record.zip_code,
                      location=house_db_record.location,
                      weather_station_id=house_db_record.weather_station_id)

        # Set the time range
        house.time_range = self._convert_epoch_times_to_range(
            house_db_record.start_epoch_time,
            house_db_record.end_epoch_time)
        return house

    def _convert_epoch_times_to_range(self, start_epoch_time: int,
                                      end_epoch_time: int) -> TimeRange:
        """
        Convert epoch times to a time range.
        """
        return TimeRange(
            start_time=epochtimems_to_datetime(
                start_epoch_time * 1000,
                timezone_str="UTC"),
            end_time=epochtimems_to_datetime(
                end_epoch_time * 1000,
                timezone_str="UTC"))

    def _load_consumption(self, house: House):
        """
        Load the consumption data for a house.
        """
        aggregator = ConsumptionAggregator(house)
        house.consumption = ConsumptionData(
            usage_10min=aggregator.get_total_consumption_10min(),
            usage_hourly=aggregator.get_total_consumption_hourly())

    def build_house_from_csv(self, house_id: int, path: str) -> House:
        """
        Build a house from a CSV file.

        The CSV file should have the following format:
        - Header: timestamp,total,appliance_1,appliance_2,...
        - Timestamp format: YYYY-MM-DD HH:MM:SS
        - Consumption values in kW

        Args:
            house_id: ID to assign to the house
            path: Path to the CSV file

        Returns:
            House: Created House instance

        Example:
            >>> house = builder.build_house_from_csv(1, "house_1.csv")
        """

        house = House(house_id)

        house.appliances = \
            ApplianceBuilder().build_appliances_from_csv(house, path)

        self._load_consumption(house)

        self._set_start_and_end_time(house)

        print(f"House {house_id} built from csv.")

        return house

    def _set_start_and_end_time(self, house: House):
        """
        Set the start and end time of a house from its consumption data.

        Args:
            house: House instance to set times for
        """
        house.time_range = TimeRange(
            start_time=list(house.consumption.usage_10min.keys())[0],
            end_time=list(house.consumption.usage_10min.keys())[-1]
        )

    def build_house_by_id(self,
                          house_id: int,
                          db_file_path: str | None = None
                          ) -> House:
        """
        Build a house from the database by its ID.

        Args:
            house_id: ID of the house to build

        Returns:
            House: Created House instance

        Raises:
            ValueError: If the house has no appliances

        Example:
            >>> house = builder.build_house_by_id(1)
        """

        if db_file_path is None:
            db_file_path = FilePathBuilder().get_irise_db_path()

        repository: HouseRepositorySQLite = HouseRepositorySQLite(
            db_path=db_file_path)
        house_db_record: HouseDbRecord = repository.get_house_record_by_id(
            house_id=house_id)

        house: House | None = self.build_house_from_db_record(
            house_db_record=house_db_record)

        if house is None:
            raise ValueError(f"House {house_id} has no appliances")
        return house


class ApplianceBuilder:
    """
    Builder for creating Appliance instances from various data sources.

    This class provides methods to create Appliance instances from:
    - Database records
    - CSV files
    """

    def __init__(self) -> None:
        """Initialize the ApplianceBuilder."""
        pass

    def build_appliances_from_csv(self, house: House, path: str
                                  ) -> list[Appliance]:
        """
        Build appliances from a CSV file.

        The CSV file should have the following format:
        - Header: timestamp,total,appliance_1,appliance_2,...
        - Timestamp format: YYYY-MM-DD HH:MM:SS
        - Consumption values in kW

        Args:
            house: Parent House instance
            path: Path to the CSV file

        Returns:
            list[Appliance]: List of created Appliance instances

        Example:
            >>> appliances = builder.build_appliances_from_csv(
            ...     house, "data.csv")
        """
        appliances: list[Appliance] = []

        repository = ApplianceRepositoryCSV(path)
        records = repository.get_appliance_records_by_house_id(house.house_id)
        types = repository.get_appliance_types()

        for record in records:

            consumption = repository.get_consumption_by_appliance_id(
                house.house_id, record.appliance_id)

            # Create the appliance
            appliance = Appliance(record.appliance_id,
                                  house,
                                  record.name,
                                  types[record.name])

            self._set_consumption(appliance, consumption)

            appliances.append(appliance)

        return appliances

    def _set_consumption(self, appliance: Appliance,
                         consumption_10min: dict[datetime, float]):
        """
        Set the consumption data for an appliance.
        This means it also creates the hourly consumption data.
        """
        # Assign the consumption data to the appliance
        # First, use the 10 min interval data
        aggregator = ConsumptionAggregatorAppliance(appliance)
        appliance.consumption = ConsumptionData(consumption_10min)

        # Then, convert the 10 min interval data to hourly intervals
        appliance.consumption.usage_hourly = \
            aggregator.get_consumption_hourly()

    def build_appliances_from_db(self, house: House,
                                 db_file_path: str) -> list[Appliance]:
        """
        Build all appliances for a house from the database.

        Excludes total site light consumption and site consumption.

        Args:
            house: Parent House instance

        Returns:
            list[Appliance]: List of created Appliance instances

        Example:
            >>> appliances = builder.build_all_appliances(house)
        """

        repository = ApplianceRepositorySQLite(db_file_path)
        records = repository.get_appliance_records_by_house_id(house.house_id)

        appliances: list[Appliance] = []
        for record in records:

            # Skip site consumption
            if record.name in ["Site consumption ()"]:
                continue

            # Build the appliance
            appliance = self.build_appliance_from_db_record(
                record, house, db_file_path)
            if appliance is not None:
                appliances.append(appliance)

        return appliances

    def build_appliance_from_db_record(self,
                                       record: ApplianceDbRecord,
                                       house: House,
                                       db_path: str | None = None
                                       ) -> Appliance | None:
        """
        Build an appliance from database record.

        Args:
            record: Appliance database record
            house: Parent House instance

        Returns:
            Appliance | None: Created Appliance instance or None if
                creation fails
        """

        inferer = ApplianceInferrer()
        appliance_type = inferer.determine_appliance_type(record.name)

        appliance = Appliance(record.appliance_id,
                              house,
                              record.name,
                              appliance_type)

        if db_path is None:
            db_path = FilePathBuilder().get_irise_db_path()

        self._load_consumption_from_db(appliance, db_path)

        return appliance

    def _load_consumption_from_db(self, appliance: Appliance,
                                  db_path: str):
        """
        Load the consumption data for an appliance.
        """
        repository = ApplianceRepositorySQLite(db_path)
        consumption_10min = repository.get_consumption_by_appliance_id(
            appliance.house.house_id,
            appliance.appliance_id)

        filtered_consumption = self._filter_outliers(consumption_10min)

        if filtered_consumption == {}:
            return None

        self._set_consumption(appliance, filtered_consumption)

    def _filter_outliers(self,
                         consumption: dict[datetime, float],
                         outlier_sensitivity: float = 10.0
                         ) -> dict[datetime, float]:
        """Filter out anomalous values in a time series
        by smoothing spikes and dips.

        Args:
            values: List of numerical values representing time series data
            outlier_sensitivity: Threshold multiplier for standard deviation
                (default: 10.0)

        Returns:
            List of values with outliers smoothed out
        """
        if len(consumption) < 3:
            return consumption.copy()

        # Calculate differences between consecutive values

        consumption_values = list(consumption.values())
        deltas = numpy.diff(consumption_values)
        mean_delta = numpy.mean(deltas)
        std_delta = numpy.std(deltas)

        # Create threshold for outlier detection
        threshold = mean_delta + outlier_sensitivity * std_delta

        # Check each point (except first and last) for outliers
        for i in range(1, len(consumption_values) - 1):
            prev_diff = consumption_values[i] - consumption_values[i - 1]
            next_diff = consumption_values[i + 1] - consumption_values[i]

            # Detect if point is a spike or dip
            is_outlier = (
                abs(prev_diff) > threshold and
                abs(next_diff) > threshold and
                prev_diff * next_diff < 0
            )

            if is_outlier:
                # Smooth the outlier by averaging neighbors
                consumption_values[i] = (consumption_values[i - 1] +
                                         consumption_values[i + 1]) / 2

        return {k: v for k, v in zip(consumption.keys(), consumption_values)}
