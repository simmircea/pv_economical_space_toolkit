"""
Utility functions and classes for the reno package.

This module provides various utilities for:
- Command line argument parsing
- Geographic location handling
- Time and space management
- File path management for data and results
"""

import argparse
from datetime import datetime, timezone
import os

import pandas as pd

import pytz

from source.constants import DATE_TIME_FORMAT


def datetime_to_stringdate(a_datetime: datetime,
                           date_format: str = DATE_TIME_FORMAT) -> str:
    return a_datetime.strftime(date_format)


def epochtimems_to_datetime(epochtimems: int, timezone_str: str
                            ) -> datetime:
    """
    Convert epoch time in milliseconds to a datetime object.
    """
    dt = datetime.fromtimestamp(epochtimems // 1000)
    localized_dt = pytz.timezone(timezone_str).localize(dt, is_dst=True)
    return localized_dt


def stringdate_to_datetime(stringdatetime: str,
                           timezone_str: str,
                           date_format: str = DATE_TIME_FORMAT) -> datetime:
    """
    Convert a string date to a datetime object.
    """
    dt = datetime.strptime(stringdatetime, date_format)
    localized_dt: datetime = pytz.timezone(
        timezone_str).localize(dt, is_dst=True)
    return localized_dt


def stringdate_to_epochtimems(
        stringdatetime: str,
        timezone_str: str,
        date_format=DATE_TIME_FORMAT,) -> int:

    dt = datetime.strptime(stringdatetime, date_format)
    localized_dt = pytz.timezone(timezone_str).localize(
        dt, is_dst=True)  # Changed is_dst to None for automatic detection
    return int(localized_dt.timestamp() * 1000)


def parse_args():
    """
    Parse command line arguments for the simulation.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - location: Simulation location (default: "Grenoble")
            - start_date: Start date in DD/MM/YYYY format
                (default: "01/3/1998")
            - end_date: End date in DD/MM/YYYY format
                (default: "01/3/1999")

    Example:
        >>> args = parse_args()
        >>> print(f"Running simulation for {args.location}")
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str,
                        required=False, default="Bucharest")
    parser.add_argument("--start_date", type=str,
                        required=False, default="01/2/1998")
    parser.add_argument("--end_date", type=str,
                        required=False, default="01/2/1999")
    parser.add_argument("--number_of_panels", type=int,
                        required=False, default=44)
    parser.add_argument("--peak_power_kW", type=float,
                        required=False, default=0.346)
    parser.add_argument("--manager_type", type=str,
                        required=False, default="reactive")
    return parser.parse_args()


def get_lat_lon_from_location(location: str) -> tuple[float, float]:
    """
    Get the latitude and longitude coordinates for a supported location.

    Args:
        location: Name of the location (currently supports "Grenoble" and
            "Bucharest")

    Returns:
        tuple[float, float]: (latitude, longitude) in decimal degrees

    Raises:
        ValueError: If the location is not supported

    Example:
        >>> lat, lon = get_lat_lon_from_location("Grenoble")
        >>> print(f"Latitude: {lat}, Longitude: {lon}")
    """
    if location == "Grenoble":
        return 45.19154994547585, 5.722065312331381
    elif location == "Bucharest":
        return 44.426827, 26.103731
    elif location == "Cayenne":
        return 4.924435336591809, -52.31276008988111
    elif location == "Paris":
        return 48.856614, 2.352222
    else:
        raise ValueError(f"Location {location} not supported")


class TimeSpaceHandler:
    """
    Handler for time and space related operations in the simulation.

    This class manages time ranges, geographic coordinates, and provides
    utilities for time-based operations in the simulation.

    Attributes:
        location: Name of the simulation location
        latitude_north_deg: Latitude in decimal degrees
        longitude_east_deg: Longitude in decimal degrees
        start_date: Start date string in DD/MM/YYYY format
        end_date: End date string in DD/MM/YYYY format
        start_time: Start datetime in UTC
        end_time: End datetime in UTC
        range_hourly: List of hourly datetimes in UTC
    """

    def __init__(self, location: str, start_date: str, end_date: str):
        """
        Initialize the TimeSpaceHandler.

        Args:
            location: Name of the simulation location
            start_date: Start date in DD/MM/YYYY format
            end_date: End date in DD/MM/YYYY format
        """
        self.location: str = location

        latitude_north_deg, longitude_east_deg = get_lat_lon_from_location(
            location)
        self.latitude_north_deg = latitude_north_deg
        self.longitude_east_deg = longitude_east_deg

        self.start_date: str = start_date
        self.end_date: str = end_date

        self.start_time_str: str = f"{start_date} 00:00:00"
        self.end_time_str: str = f"{end_date} 00:00:00"

        self.start_time: datetime = stringdate_to_datetime(
            self.start_time_str, timezone_str="UTC")  # type: ignore
        self.end_time: datetime = stringdate_to_datetime(
            self.end_time_str, timezone_str="UTC")  # type: ignore

        self.start_epochtimems: int = stringdate_to_epochtimems(
            self.start_time_str, date_format='%d/%m/%Y %H:%M:%S',
            timezone_str="UTC")
        self.end_epochtimems: int = stringdate_to_epochtimems(
            self.end_time_str, date_format='%d/%m/%Y %H:%M:%S',
            timezone_str="UTC")

        self._set_time_range()

    def _set_time_range(self):
        """
        Set the time range based on the start and end time.

        Creates an hourly range of datetimes, removing the DST transition
        hour to match consumption and production data. All times are
        converted to UTC.
        """
        # Create hourly range directly
        hourly_dates = pd.date_range(
            self.start_time, self.end_time, freq="h", inclusive="both"
        )

        # Convert to naive datetimes and remove DST transition hour
        self.range_hourly: list[datetime] = [
            dt.replace(tzinfo=None) for dt in hourly_dates
            if not (dt.month == 3 and dt.day >= 25 and dt.day <= 31
                    and dt.hour == 3 and dt.weekday() == 6)
        ]

        # Convert to UTC to match the consumption and production data
        self.range_hourly = [dt.replace(tzinfo=timezone.utc)
                             for dt in self.range_hourly]

    def get_k_from_datetime(self, datetime: datetime) -> int:
        """
        Get the index k for a given datetime in the hourly range.

        Args:
            datetime: The datetime to find the index for

        Returns:
            int: Index of the datetime in the hourly range

        Raises:
            ValueError: If the datetime is not in the range
        """
        return self.range_hourly.index(datetime)

    def get_datetime_from_k(self, k: int) -> datetime:
        """
        Get the datetime at index k in the hourly range.

        Args:
            k: Index in the hourly range

        Returns:
            datetime: The datetime at index k

        Raises:
            IndexError: If k is out of range
        """
        return self.range_hourly[k]

    def get_start_time_str(self) -> str:
        return self.start_date.replace("/", "_")

    def get_end_time_str(self) -> str:
        return self.end_date.replace("/", "_")

    def get_key(self) -> str:
        """
        Get the key of the time space handler.
        """
        start_time = self.get_start_time_str()
        end_time = self.get_end_time_str()
        return f"{self.location}_{start_time}_{end_time}"

    def to_json(self) -> dict:
        """
        Convert the time space handler to a JSON dictionary.
        """
        return {
            "location": self.location,
            "start_date": self.start_date,
            "end_date": self.end_date
        }


class FilePathBuilder:
    """
    Builder for file paths used in the simulation.

    This class provides methods to generate consistent file paths for
    various data files, results, and plots used in the simulation.
    """

    def __init__(self):
        """Initialize the FilePathBuilder."""
        pass

    def get_data_folder(self) -> str:
        """
        Get the path to the data folder.
        """
        folder_name = "data"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        return folder_name

    def get_community_data_folder(self) -> str:
        """
        Get the path to the community data folder.
        """
        data_folder = self.get_data_folder()
        folder_name = os.path.join(data_folder, "community")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        return folder_name

    def get_irise_db_path(self) -> str:
        """
        Get the path to the IRISE database.

        Returns:
            str: Path to the IRISE database file
        """
        data_folder = self.get_data_folder()
        return os.path.join(data_folder, "irise38.sqlite3")

    def get_results_folder(self) -> str:
        """
        Get the path to the results folder.
        """
        folder_name = "results"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        return folder_name

    def get_plots_folder(self) -> str:
        """
        Get the path to the plots folder, creating it if it doesn't exist.

        Returns:
            str: Path to the plots folder
        """
        folder_name = "plots"
        path = os.path.join("batem", "reno", folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_community_valid_houses_path(
            self, time_space_handler: TimeSpaceHandler) -> str:
        """
        Get the path to the community valid houses file.

        Args:
            time_space_handler: TimeSpaceHandler instance for time range

        Returns:
            str: Path to the valid houses JSON file
        """
        file_name = (f"community_valid_houses_{time_space_handler.location}_"
                     f"{time_space_handler.start_date.replace('/', '_')}_"
                     f"{time_space_handler.end_date.replace('/', '_')}.json")
        community_data_folder = self.get_community_data_folder()
        return os.path.join(community_data_folder, file_name)
