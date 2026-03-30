
from datetime import datetime, timedelta
import os
from batem.core import weather
import pandas as pd
import pytz
from batem.core import solar
from batem.core.weather import SWDbuilder, SiteWeatherData


from source.utils import datetime_to_stringdate, stringdate_to_datetime
from source.pv.model import (
    PVConfig, PVPlant, ProductionData)
from source.utils import FilePathBuilder, TimeSpaceHandler


def conver_pv_datetime_to_utc(
        datetimes: list[datetime]) -> list[datetime]:
    """
    Convert the PV datetimes to UTC.
    """
    return [dt.replace(tzinfo=pytz.timezone("UTC"))
            for dt in datetimes]


class WeatherDataBuilder:
    def __init__(self):
        pass

    def build(self, ts_handler: TimeSpaceHandler):
        """
        The location is a string like "Paris, France".
        The latitude_north_deg and longitude_east_deg
        are the coordinates of the location.
        The from_datetime_string and to_datetime_string
        are the dates of the data to be fetched.
        The from_datetime_string and to_datetime_string
        are in the format "DD/MM/YYYY HH:MM:SS".
        The from_date and to_date are the dates of the data to be fetched.
        The from_date and to_date are in the format "YYYY-MM-DD".
        """
        # -- Extract only the date, as per requirements of the openw- API
        from_date = ts_handler.start_date.split(" ")[0]
        to_date = ts_handler.end_date.split(" ")[0]

        # to_datetime_str = self._adapt_end_time(to_date)
        # to_date = to_datetime_str.split(" ")[0]

        print(f"From date: {from_date}")
        print(f"To date: {to_date}  ")

        print(
            f"Extracting weather data from {from_date} to {to_date} "
            f"for location {ts_handler.location} "
            f"(latitude: {ts_handler.latitude_north_deg}, "
            f"longitude: {ts_handler.longitude_east_deg})")

        site_weather_data = SWDbuilder(
            location=ts_handler.location,
            latitude_north_deg=ts_handler.latitude_north_deg,
            longitude_east_deg=ts_handler.longitude_east_deg
        )(from_stringdate=from_date, to_stringdate=to_date)

        return site_weather_data

    def _adapt_end_time(self, end_date: str) -> str:
        """
        Adapt the end date and substract a day because
        weather data includes the next day.
        """
        # Clean up any extra spaces
        end_date = end_date.strip()

        # If the date doesn't have time, add it
        if " " not in end_date:
            end_date = f"{end_date} 00:00:00"

        # Parse the datetime
        end_date_as_datetime = stringdate_to_datetime(
            end_date,
            timezone_str="UTC"
        )

        # Subtract a day and convert back to string
        new_end_date = datetime_to_stringdate(
            end_date_as_datetime - timedelta(days=1)  # type: ignore
        )
        return new_end_date


class PVPlantBuilder:
    def __init__(self):
        pass

    def build_from_solar_model(self,
                               weather_data: weather.SiteWeatherData,
                               solar_model: solar.SolarModel,
                               config: PVConfig):
        """
        Build a PV plant from a solar model.
        """
        pv_plant = solar.PVplant(
            solar_model,
            exposure_deg=config.exposure_deg,
            slope_deg=config.slope_deg,
            mount_type=config.mount_type,
            peak_power_kW=config.get_peak_power_kW(),
            number_of_panels=config.number_of_panels,
            panel_height_m=config.panel_height_m,
            panel_width_m=config.panel_width_m,
            pv_efficiency=config.pv_efficiency,
            temperature_coefficient=config.temperature_coefficient,
            distance_between_arrays_m=config.distance_between_arrays_m)

        # Convert the datetimes to UTC
        datetimes = conver_pv_datetime_to_utc(
            weather_data.datetimes)

        # Convert the powers to kW
        power_production = self._build_production_data(
            datetimes, pv_plant.powers_W())

        self._add_zero_value_for_first_hour(power_production)

        plant = PVPlant(
            weather_data=weather_data,
            solar_model=solar_model,
            config=config)

        plant.production = ProductionData(usage_hourly=power_production)
        return plant

    def build(self, weather_data: SiteWeatherData,
              config: PVConfig):

        solar_model = solar.SolarModel(weather_data)

        pv_plant = self.build_from_solar_model(
            weather_data, solar_model, config)

        return pv_plant

    def _build_production_data(
        self, datetimes: list[datetime],
        powers_W: list[float] | dict[str,
                                     dict[str, list[float]]]
    ) -> dict[datetime, float]:
        """
        Build the production data for a PV plant.

        datetimes: list[datetime] as UTC
        powers_W: list[float] as W
        """
        power_production = {
            timestamp: float(production)/1000
            for timestamp, production in zip(datetimes, powers_W)}
        return power_production

    def _add_zero_value_for_first_hour(
            self,
            power_production: dict[datetime, float]):
        """
        Add a zero value for the first hour.
        """
        start_of_range = list(power_production.keys())[0]
        new_start = start_of_range - pd.Timedelta(hours=1)
        power_production[new_start] = 0  # type: ignore


class PVPlantFilePathBuilder:

    def __init__(self, pv_config: PVConfig,
                 time_space_handler: TimeSpaceHandler):
        self._pv_config = pv_config
        self._time_space_handler = time_space_handler
        self._file_path_builder = FilePathBuilder()

    def get_pv_plant_path(self) -> str:
        """
        Get the path to the PV plant data.

        Args:
            time_space_handler: TimeSpaceHandler instance for time range

        Returns:
            str: Path to the PV plant data file
        """
        start_date = self._time_space_handler.start_date
        end_date = self._time_space_handler.end_date
        location = self._time_space_handler.location
        peak_power_kW = self._pv_config.get_peak_power_kW()
        file_name = (f"pv_plant_{location}_{start_date.replace('/', '_')}_"
                     f"{end_date.replace('/', '_')}_{peak_power_kW} kW.csv")

        folder = self._file_path_builder.get_data_folder()
        return os.path.join(folder, file_name)


class PVSizingFilePathBuilder:
    def __init__(self):
        self._file_path_builder = FilePathBuilder()

    def get_pv_sizing_folder(self) -> str:
        """
        Get the path to the PV sizing folder.
        """
        results_folder = self._file_path_builder.get_results_folder()
        folder = os.path.join(results_folder, "sizing")
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder

    def get_pv_statistics_csv_path(self) -> str:
        """
        Get the path to the PV statistics CSV file.
        """
        folder = self.get_pv_sizing_folder()
        file_name = "pv_statistics.csv"
        return os.path.join(folder, file_name)

    def get_economical_space_csv_path(self, indicator_name: str) -> str:
        """
        Get the path to the economical space CSV file.
        """
        folder = self.get_pv_sizing_folder()
        file_name = f"economical_space_{indicator_name}.csv"
        return os.path.join(folder, file_name)
