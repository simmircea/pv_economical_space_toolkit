from dataclasses import dataclass
from datetime import datetime
from source.house.constants import ACTIONABLE_BY_PRESENCE, MONTH_TO_SEASON, SHIFTABLE_APPLIANCES
from source.house.model import Appliance


@dataclass
class RunProfile:
    """
    The profile including attributes of a run.

    Attributes:
        day_of_week: The day of the week (0-6, 0=Monday, 6=Sunday)
        start_time_hour: The hour of the day when the run starts
        end_time_hour: The hour of the day when the run ends
        duration_mins: The duration of the run in minutes
        season: The season of the run (winter, spring, summer, autumn)
        shiftable: Whether the run can be shifted in time
        actionable_by_presence: Whether the run can be actioned exclusively
        by the presence of the user
    """
    day_of_week: int
    start_time_hour: int
    end_time_hour: int
    duration_mins: int
    season: str
    shiftable: bool
    actionable_by_presence: bool


@dataclass
class Run:
    """
    A run is a period of time when the appliance is on.
    """
    start_time: datetime
    end_time: datetime
    timestamps: list[datetime]
    load_profile_kw: list[float]
    profile: RunProfile | None = None


class ApplianceProcessor:
    """
    This is a class to process appliance consumption data
    and partition it relative to days or other time periods.
    """

    def __init__(self, appliance: Appliance):
        self.appliance = appliance

    def process_consumption(self):
        """
        Process the consumption data for the appliance.
        Partition the consumption data by day, month and year.
        Returns a dictionary with (day, month, year) as the key
        and a list of (timestamp, load_kw) as the value.
        The consumption data is stored in kW.
        """
        usage = self.appliance.consumption.usage_10min
        load_by_day: dict[tuple[int, int, int],
                          list[tuple[datetime, float]]] = {}
        for timestamp, load_kw in usage.items():
            key = (timestamp.day, timestamp.month, timestamp.year)
            if key not in load_by_day:
                load_by_day[key] = []
            load_by_day[key].append((timestamp, load_kw))
        return load_by_day

    def determine_on_profile(self, load_by_day: dict[
            tuple[int, int, int], list[tuple[datetime, float]]]):
        """
        Determine if the appliance is on or off for each sample.
        Returns a dictionary with (day, month, year) as the key
        and a list of booleans (True=on) as the value.
        """
        on_profile_by_day: dict[tuple[int, int, int], list[bool]] = {
            key: [load_kw > 0 for _, load_kw in samples]
            for key, samples in load_by_day.items()
        }
        return on_profile_by_day

    def determine_runs(
            self,
            load_by_day: dict[tuple[int, int, int],
                              list[tuple[datetime, float]]],
            on_profile_by_day: dict[tuple[int, int, int],
                                    list[bool]],
            with_profile: bool = False) -> list[Run]:
        """
        Determine the runs for the appliance.
        A run is a period of time when the appliance is on.
        """
        runs: list[Run] = []

        for key, samples in load_by_day.items():
            on_profile = on_profile_by_day[key]

            if len(on_profile) != len(samples):
                raise ValueError(
                    "load_by_day and on_profile_by_day "
                    "must have the same length for each key")

            i = 0
            n = len(on_profile)

            while i < n:
                # Skip periods where the appliance is off
                if not on_profile[i]:
                    i += 1
                    continue

                # We are at the start of a run
                start_idx = i
                while i < n and on_profile[i]:
                    i += 1
                end_idx = i  # first index after the run

                start_time = samples[start_idx][0]
                end_time = samples[end_idx - 1][0]
                segment = samples[start_idx:end_idx]
                timestamps = [ts for ts, _ in segment]
                load_profile_kw = [load_kw for _, load_kw in segment]

                if with_profile:

                    shiftable = self.appliance.type in SHIFTABLE_APPLIANCES
                    actionable_by_presence = (
                        self.appliance.type in ACTIONABLE_BY_PRESENCE
                    )
                    profile = RunProfile(
                        day_of_week=start_time.weekday(),
                        start_time_hour=start_time.hour,
                        end_time_hour=end_time.hour,
                        duration_mins=int(
                            (end_time - start_time).total_seconds() / 60),
                        season=MONTH_TO_SEASON[start_time.month],
                        shiftable=shiftable,
                        actionable_by_presence=actionable_by_presence)
                else:
                    profile = None

                runs.append(
                    Run(start_time=start_time,
                        end_time=end_time,
                        timestamps=timestamps,
                        load_profile_kw=load_profile_kw,
                        profile=profile))

        return runs
