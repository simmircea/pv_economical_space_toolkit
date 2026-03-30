from abc import ABC, abstractmethod
import csv
from datetime import datetime
# from typing import override

import pandas as pd

from source.utils import epochtimems_to_datetime
from source.house.db import execute_query, get_db_connection
from source.house.constants import APPLIANCES
from source.house.model import (ApplianceDbRecord, HouseDbRecord,
                                WeatherStationDbRecord)


class ApplianceRepository(ABC):
    """Abstract interface for appliance data access.
    The repository is used to get the appliance records and consumption data.
    """

    @abstractmethod
    def get_appliance_records_by_house_id(self, house_id: int
                                          ) -> list[ApplianceDbRecord]:
        """Get all appliance records for a specific house."""
        pass

    @abstractmethod
    def get_consumption_by_appliance_id(self, house_id: int,
                                        appliance_id: int
                                        ) -> dict[datetime, float]:
        """Get consumption by appliance ID."""
        pass


class HouseRepository(ABC):
    """Abstract interface for house data access.
    The repository is used to get the house records.
    """

    @abstractmethod
    def get_all_house_records(self) -> list[HouseDbRecord]:
        """Get all house records from data source."""
        pass

    @abstractmethod
    def get_house_record_by_id(self, house_id: int) -> HouseDbRecord | None:
        """Get specific house record by ID."""
        pass


class WeatherStationRepository(ABC):
    """Abstract interface for weather station data access."""

    @abstractmethod
    def get_weather_station_record_by_id(
            self, weather_station_id: int
            ) -> WeatherStationDbRecord:
        """Get specific weather station record by ID."""
        pass


class ApplianceRepositorySQLite(ApplianceRepository):
    """SQLite implementation of the appliance repository."""

    def __init__(self, db_path: str) -> None:
        """Initialize the appliance repository
        with the database SQLite path.
        """
        self.db_path: str = db_path

    def get_appliance_records_by_house_id(self,
                                          house_id: int
                                          ) -> list[ApplianceDbRecord]:
        """Get all appliance records for a specific house."""

        query = """
        SELECT ID, HouseIDREF, Name
        FROM Appliance
        WHERE HouseIDREF = ?
        """
        appliances_data = execute_query(query, (house_id,))

        return [ApplianceDbRecord(
                appliance_id=appliance_data[0],
                house_id=appliance_data[1],
                name=appliance_data[2])
                for appliance_data in appliances_data]

    def get_consumption_by_appliance_id(self,
                                        house_id: int,
                                        appliance_id: int
                                        ) -> dict[datetime, float]:
        """
        Get the consumption data for an appliance with optimized performance.
        The consumption data is stored in 10-minute intervals.
        The data is stored in kWh.
        """

        query = """
        SELECT EpochTime, Value
        FROM Consumption
        WHERE ApplianceIDREF = ?
        AND HouseIDREF = ?
        ORDER BY EpochTime
        """

        with get_db_connection() as conn:
            # Enable WAL mode for better read performance
            _ = conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            _ = cursor.execute(query, (appliance_id, house_id))

            # Use list comprehension for faster data processing
            consumption_data_dict = {
                epochtimems_to_datetime(
                    int(epoch) * 1000,  # convert to milliseconds
                    timezone_str="UTC"): float(value)/1000  # to kW
                for epoch, value in cursor.fetchall()
            }

        return consumption_data_dict


class ApplianceRepositoryCSV(ApplianceRepository):
    """CSV implementation of the appliance repository."""

    def __init__(self, csv_path: str):
        """Initialize the appliance repository
        with the CSV path.
        """
        self.csv_path: str = csv_path
        self._header_cache: list[str] | None = None
        self._appliance_data_cache: dict[int, dict[datetime, float]] = {}

    def _get_header(self) -> list[str]:
        """Get CSV header (cached)."""
        if self._header_cache is None:
            with open(self.csv_path, 'r') as f:
                reader = csv.reader(f)
                self._header_cache = next(reader)
        return self._header_cache

    def get_appliance_records_by_house_id(self, house_id: int
                                          ) -> list[ApplianceDbRecord]:
        """Get all appliance records for a specific house."""
        header = self._get_header()

        records: list[ApplianceDbRecord] = []
        for key in header:
            if key in ['total', 'timestamp']:
                continue

            try:
                appliance_id, name, _ = key.split('_')
                appliance_id = int(appliance_id)

                record = ApplianceDbRecord(
                    appliance_id=appliance_id,
                    house_id=house_id,
                    name=name
                )
                records.append(record)
            except ValueError:
                # Skip malformed column names
                continue

        return records

    def get_consumption_by_appliance_id(self,
                                        house_id: int,
                                        appliance_id: int
                                        ) -> dict[
            datetime, float]:
        """Get consumption by appliance ID (lazy loaded)."""

        if appliance_id not in self._appliance_data_cache:
            # Load only when requested
            df = pd.read_csv(self.csv_path, index_col=0, parse_dates=[0])

            # Find the column for this appliance
            target_column = None
            for col in df.columns:
                if col in ['total', 'timestamp']:
                    continue
                try:
                    col_appliance_id, _, _ = col.split('_')
                    if int(col_appliance_id) == appliance_id:
                        target_column = col
                        break
                except ValueError:
                    continue

            if target_column is None:
                raise ValueError(
                    f"Appliance {appliance_id} not found in CSV file")

            # Cache this appliance's data
            self._appliance_data_cache[appliance_id] = \
                df[target_column].to_dict(
            )

        return self._appliance_data_cache[appliance_id]

    def get_appliance_types(self) -> dict[str, APPLIANCES]:
        """Get all appliance types."""
        header = self._get_header()

        appliance_types: dict[str, APPLIANCES] = {}
        for key in header:
            if key in ['total', 'timestamp']:
                continue

            try:
                _, name, appliance_type = key.split('_')
                appliance_types[name] = APPLIANCES(appliance_type)
            except ValueError:
                # Skip malformed column names or unknown types
                continue

        return appliance_types


class HouseRepositorySQLite(HouseRepository):
    """SQLite implementation of the house repository."""

    def __init__(self, db_path: str):
        """Initialize the house repository
        with the database SQLite path.
        """
        self.db_path: str = db_path

    def get_all_house_records(self) -> list[HouseDbRecord]:
        """Get all house records from data source."""
        query = """
        SELECT ID, ZIPcode, Location, WeatherStationIDREF, \
            StartingEpochTime, EndingEpochTime
        FROM House
        """
        houses_data = execute_query(query)

        return [HouseDbRecord(
                house_id=house_data[0],
                zip_code=house_data[1],
                location=house_data[2],
                weather_station_id=house_data[3],
                start_epoch_time=house_data[4],
                end_epoch_time=house_data[5]
                )
                for house_data in houses_data]

    def get_house_record_by_id(self, house_id: int
                               ) -> HouseDbRecord:
        """Get specific house record by ID."""
        query = """
        SELECT ID, ZIPcode, Location, WeatherStationIDREF, \
            StartingEpochTime, EndingEpochTime
        FROM House
        WHERE ID = ?
        """
        house_data = execute_query(query, (house_id,))

        if len(house_data) == 0:
            raise ValueError(f"House {house_id} not found in database")

        return HouseDbRecord(
            house_id=house_data[0][0],
            zip_code=house_data[0][1],
            location=house_data[0][2],
            weather_station_id=house_data[0][3],
            start_epoch_time=house_data[0][4],
            end_epoch_time=house_data[0][5])


class WeatherStationRepositorySQLite(WeatherStationRepository):
    """SQLite implementation of the weather station repository."""

    def __init__(self, db_path: str):
        """Initialize the weather station repository
        with the database SQLite path.
        """
        self.db_path: str = db_path

    def get_weather_station_record_by_id(
            self, weather_station_id: int
            ) -> WeatherStationDbRecord:
        """Get specific weather station record by ID."""
        query = """
        SELECT ID, Location, Latitude, Longitude
        FROM WeatherStation
        WHERE ID = ?
        """
        station_data = execute_query(query, (weather_station_id,))

        if len(station_data) == 0:
            raise ValueError(
                f"Weather station {weather_station_id} "
                "not found in database")

        row = station_data[0]
        return WeatherStationDbRecord(
            weather_station_id=row[0],
            location=row[1],
            latitude=row[2],
            longitude=row[3])