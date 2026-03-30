import csv
import os
from typing import Optional

from batem.core import weather
from source.house.constants import APPLIANCES, location_to_city
from source.house.db import execute_query
from source.house.model import (
    House, Appliance, TimeRange, WeatherStationDbRecord)
import pandas as pd
from datetime import datetime

from source.pv.creation import PVConfig, PVPlantBuilder, WeatherDataBuilder
from source.pv.model import PVPlant
from source.utils import FilePathBuilder, TimeSpaceHandler


class HouseFilePathBuilder:
    """
    This class builds the paths to the house consumption data.
    """

    def __init__(self):
        self.file_path_builder = FilePathBuilder()

    def get_csv_data_folder(self) -> str:
        """
        Get the path to the csv data folder.
        """
        data_folder = self.file_path_builder.get_data_folder()
        csv_data_folder = os.path.join(data_folder, "csv_data")
        if not os.path.exists(csv_data_folder):
            os.makedirs(csv_data_folder)
        return csv_data_folder

    def get_weather_station_mapping_csv_path(self) -> str:
        """
        Get the path to the weather station mapping CSV file.
        """
        csv_data_folder = self.get_csv_data_folder()
        return os.path.join(csv_data_folder, "weather_station_mapping.csv")

    def get_valid_houses_csv_path(self) -> str:
        """
        Get the path to the valid houses CSV file.
        """
        csv_data_folder = self.get_csv_data_folder()
        return os.path.join(csv_data_folder, "valid_houses.csv")

    def get_trimmed_house_consumption_csv_path(
            self, house_id: int,
            time_space_handler: TimeSpaceHandler,
            hourly: bool = False) -> str:
        """
        Get the path to the trimmed house consumption data.

        Args:
            house_id: ID of the house
            time_space_handler: TimeSpaceHandler instance for time range
            hourly: Whether to use hourly data (default: False)

        Returns:
            str: Path to the trimmed consumption data file
        """
        start_time = time_space_handler.get_start_time_str()
        end_time = time_space_handler.get_end_time_str()
        if hourly:
            file_name = f"{house_id}_consumption_hourly_trimmed_{start_time}_"
            file_name = f"{file_name}_{end_time}.csv"
        else:
            file_name = f"{house_id}_consumption_trimmed_{start_time}_"
            file_name = f"{file_name}_{end_time}.csv"

        csv_data_folder = self.get_csv_data_folder()
        return os.path.join(csv_data_folder, file_name)

    def get_house_consumption_csv_path(self, house_id:
                                       int, hourly: bool = False) -> str:
        """
        Get the path to the house consumption data.

        Args:
            house_creation_experiment: HouseCreationExperiment instance
            hourly: Whether to use hourly data (default: False)

        Returns:
            str: Path to the consumption data file
        """
        if hourly:
            file_name = f"{house_id}_consumption_hourly.csv"
        else:
            file_name = f"{house_id}_consumption.csv"

        csv_data_folder = self.get_csv_data_folder()

        return os.path.join(csv_data_folder, file_name)

    def get_results_folder(self, house_id: int) -> str:
        """
        Get the path to the results folder.
        """
        folder = self.file_path_builder.get_results_folder()
        dir = f"house_{house_id}_results"
        if not os.path.exists(os.path.join(folder, dir)):
            os.makedirs(os.path.join(folder, dir))
        return os.path.join(folder, dir)


class ApplianceInferrer:
    def __init__(self):
        pass

    def determine_appliance_type(self, name: str) -> APPLIANCES:
        """
        Infer the type of an appliance from the name.
        """
        inferred_type: APPLIANCES | None = None
        for appliance_type in APPLIANCES:
            if appliance_type.name.lower() in name.lower():
                inferred_type = appliance_type
                break
        if inferred_type is None:
            inferred_type = APPLIANCES.OTHER
        return inferred_type


class ConsumptionAggregatorAppliance:
    def __init__(self, appliance: Appliance):
        self.appliance = appliance

    def get_consumption_hourly(self) -> dict[datetime, float]:
        """
        Convert 10-minute consumption data to hourly intervals.

        Uses floor('h') instead of resample('h') to prevent creating
        extra timestamps during DST transitions.

        Returns:
            dict[datetime, float]: Dictionary of hourly consumption data
        """
        consumption_series = pd.Series(self.appliance.consumption.usage_10min)
        hourly_consumption: pd.Series = consumption_series.groupby(
            consumption_series.index.floor('h')  # type: ignore
        ).sum()
        return hourly_consumption.to_dict()


class ConsumptionTrimmer:
    def __init__(self, house: House):
        self.house = house

    def trim_consumption_house(self, time_space_handler: TimeSpaceHandler):
        self.house.time_range = TimeRange(
            start_time=time_space_handler.start_time,
            end_time=time_space_handler.end_time
        )
        for appliance in self.house.appliances:
            self.trim_consumption_appliance(appliance, time_space_handler)

    def trim_consumption_appliance(self, appliance: Appliance,
                                   time_space_handler: TimeSpaceHandler):
        """
        Trim consumption data to the specified time range.

        Args:
            time_space_handler: TimeSpaceHandler instance defining the
                time range to trim to
        """
        new_range = TimeRange(
            start_time=time_space_handler.start_time,
            end_time=time_space_handler.end_time
        )

        appliance.consumption.usage_10min = {
            k: v for k, v in appliance.consumption.usage_10min.items()
            if new_range.contains(k)}


class ConsumptionAggregator:
    def __init__(self, house: House):
        self.house = house

    def get_total_consumption_hourly(self) -> dict[datetime, float]:
        """
        Aggregates consumption from all appliances at hourly intervals.

        Returns a dictionary with the total consumption for each hour.
        """
        # Get 10-minute consumption
        total_consumption_10min = self.get_total_consumption_10min()

        # Get hourly consumption by properly grouping 10-minute data
        # This properly aggregates all 10-minute readings within each hour
        # We do this because resmapling adds the timestamp for DST transitions
        consumption_series = pd.Series(total_consumption_10min)
        hourly_consumption: pd.Series = consumption_series.groupby(
            consumption_series.index.floor('h')  # type: ignore
        ).sum()

        return hourly_consumption.to_dict()

    def get_total_consumption_10min(self) -> dict[datetime, float]:
        """
        Aggregates consumption from all appliances at 10-minute intervals.

        Returns a dictionary with the total
        consumption for each 10-minute interval.
        """

        appliances_consumption_10min = [
            appliance.consumption.usage_10min
            for appliance in self.house.appliances
        ]

        # Get all unique timestamps
        all_timestamps = set()
        for consumption_in_time in appliances_consumption_10min:
            all_timestamps.update(consumption_in_time.keys())

        # Sum consumption for each timestamp
        return {
            timestamp: sum(
                consumption_in_time.get(timestamp, 0.0)
                for consumption_in_time in appliances_consumption_10min
            )
            for timestamp in sorted(all_timestamps)
        }


class ConsumptionExporter:
    def __init__(self, house: House):
        self.house = house
        self.house_path_builder = HouseFilePathBuilder()

    def get_house_consumption(self, hourly: bool = False
                              ) -> dict[datetime, float]:
        aggregator = ConsumptionAggregator(self.house)
        if hourly:
            return aggregator.get_total_consumption_hourly()
        else:
            return aggregator.get_total_consumption_10min()

    def get_appliances_consumption(self, appliances: list[Appliance],
                                   hourly: bool = False
                                   ) -> dict[str, dict[datetime, float]]:
        consumption_by_appliance = {}

        for appliance in appliances:
            aggregator = ConsumptionAggregatorAppliance(appliance)
            if hourly:
                consumption_by_appliance[appliance.name] = \
                    aggregator.get_consumption_hourly()
            else:
                consumption_by_appliance[appliance.name] = \
                    appliance.consumption.usage_10min
        return consumption_by_appliance

    def to_csv(self, path: Optional[str] = None, hourly: bool = False):
        """
        Save the house consumption data to a CSV file.

        The CSV file will have the following format:
        - Header: timestamp,total,appliance_1,appliance_2,...
        - Timestamp format: YYYY-MM-DD HH:MM:SS
        - Consumption values in kW

        Args:
            path: Path to save the CSV file
            hourly: Whether to use hourly data (default: False)

        Example:
            >>> house.to_csv("house_1_consumption.csv", hourly=True)
        """
        consumption_by_time = self.get_house_consumption(hourly)
        consumption_by_appliance = self.get_appliances_consumption(
            self.house.appliances, hourly)

        if path is None:
            path = self.house_path_builder.get_house_consumption_csv_path(
                self.house.house_id, hourly)

        with open(path, "w") as f:
            writer = csv.writer(f)

            self._write_header(writer)

            # Write the first part with the aggregated consumption
            for timestamp, consumption in consumption_by_time.items():
                result = {'timestamp': timestamp, 'total': consumption}

                # Iterate over all appliances and add their consumption
                # to the row
                for appliance in self.house.appliances:
                    result[appliance.name] = \
                        consumption_by_appliance[appliance.name][timestamp]

                writer.writerow(result.values())

        print(f"House {self.house.house_id} saved to csv.")

    def _write_header(self, writer: csv.writer):  # type: ignore
        header = ["timestamp", "total"]
        for appliance in self.house.appliances:
            key = f'{appliance.appliance_id}_{appliance.name}_{
                appliance.type.value}'
            header.append(key)
        writer.writerow(header)

    def get_appliance_consumption(
            self, appliance: Appliance,
            timestamp: datetime,
            hourly: bool = False) -> float:
        """
        Get the consumption for an appliance at a given timestamp.
        """
        if hourly:
            hourly_consumption = ConsumptionAggregatorAppliance(
                appliance).get_consumption_hourly()
            if timestamp in hourly_consumption:
                appliance_consumption = \
                    hourly_consumption[timestamp]
            else:
                appliance_consumption = 0.0
        else:
            if timestamp in appliance.consumption.usage_10min:
                appliance_consumption = \
                    appliance.consumption.usage_10min[timestamp]
            else:
                appliance_consumption = 0.0
        return appliance_consumption


def export_house_weather_mapping() -> str:
    """
    Export a CSV mapping houses to their weather stations.

    The CSV will contain, for each house:
        - house_id
        - zip_code
        - house_location
        - weather_station_id
        - weather_station_location
        - latitude
        - longitude

    Returns:
        str: Path to the generated CSV file.
    """
    builder = HouseFilePathBuilder()
    csv_path = builder.get_weather_station_mapping_csv_path()

    query = """
    SELECT h.ID AS house_id,
           h.ZIPcode AS zip_code,
           h.Location AS house_location,
           h.WeatherStationIDREF AS weather_station_id,
           ws.Location AS weather_station_location,
           ws.Latitude AS latitude,
           ws.Longitude AS longitude
    FROM House h
    LEFT JOIN WeatherStation ws
        ON h.WeatherStationIDREF = ws.ID
    ORDER BY h.ID
    """

    rows = execute_query(query)

    header = [
        "house_id",
        "zip_code",
        "house_location",
        "weather_station_id",
        "weather_station_location",
        "latitude",
        "longitude",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    return csv_path


def load_house_weather_mapping() -> dict[int, WeatherStationDbRecord]:
    """
    Load the house weather mapping from the CSV file.
    """
    csv_path = export_house_weather_mapping()
    mapping = {}
    with open(csv_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            house_id = int(row[0])
            weather_station_id = int(row[3])
            weather_station_location = row[4]
            latitude = float(row[5])
            longitude = float(row[6])
            mapping[house_id] = WeatherStationDbRecord(
                weather_station_id=weather_station_id,
                location=weather_station_location,
                latitude=latitude,
                longitude=longitude)
    return mapping


def export_valid_houses_to_csv(house_to_city: dict[int, str]):
    """
    Export the valid houses to a CSV file.
    """
    csv_path = HouseFilePathBuilder().get_valid_houses_csv_path()
    with open(csv_path, "w") as csv_file:
        writer = csv.writer(csv_file)
        header = ["house_id", "city"]
        writer.writerow(header)
        for house_id, city in house_to_city.items():
            writer.writerow([house_id, city])


def load_valid_houses_from_csv() -> dict[int, str]:
    """
    Load the valid houses from a CSV file.
    """
    csv_path = HouseFilePathBuilder().get_valid_houses_csv_path()
    house_to_city = {}
    with open(csv_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            house_id = int(row[0])
            city = row[1]
            house_to_city[house_id] = city
    return house_to_city


def return_house_ids_if_weather_station_exists() -> dict[int, str]:
    """
    Return the house IDs if the weather station exists.
    """
    mapping = load_house_weather_mapping()
    city_by_house_id = {}
    for house_id, weather_station_record in mapping.items():
        if weather_station_record.location in location_to_city:
            weather_location = weather_station_record.location
            city_by_house_id[house_id] = location_to_city[weather_location]
    return city_by_house_id


def build_pv_plant_for_house(
    house: House,
    ts_handler: TimeSpaceHandler,
    pv_config: PVConfig
) -> tuple[House, weather.SiteWeatherData, PVPlant]:
    """
    Trim the consumption of the house to the given time space handler.
    Extract the weather data for the given time space handler.
    Build the PV plant for the given time space handler and configuration.
    """

    ConsumptionTrimmer(house).trim_consumption_house(ts_handler)
    house.consumption.usage_hourly = ConsumptionAggregator(
        house).get_total_consumption_hourly()

    weather_data = WeatherDataBuilder().build(ts_handler)

    pv_plant = PVPlantBuilder().build(weather_data=weather_data,
                                      config=pv_config)

    return house, weather_data, pv_plant
