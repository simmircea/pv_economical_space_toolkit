from typing import TYPE_CHECKING, Optional
import csv
from source.house.model import House
from source.pv.model import PVPlant

from source.indicators.models import (
    BatteryIndicators, NPVConfig, avg_battery_variation, battery_protection,
    BasicIndicators, cost, neeg, npv, self_consumption, self_sufficiency)

if TYPE_CHECKING:
    from source.battery.model import Battery


def calculate_basic_indicators(house: House, pv_plant: PVPlant,
                               battery: Optional['Battery'] = None
                               ) -> BasicIndicators:
    if not house.consumption.usage_hourly:
        raise ValueError("Consumption is not set")
    if not pv_plant.production.usage_hourly:
        raise ValueError("Production is not set")

    battery_power_by_time = (
        battery.get_battery_power_by_time() if battery else {})
    consumption_by_time = house.consumption.usage_hourly
    production_by_time = pv_plant.production.usage_hourly

    neeg_value = neeg(
        consumption_by_time,
        production_by_time,
        battery_power_by_time=battery_power_by_time)
    sc_value = self_consumption(
        consumption_by_time,
        production_by_time,
        battery_power_by_time=battery_power_by_time)
    ss_value = self_sufficiency(consumption_by_time,
                                production_by_time,
                                battery_power_by_time=battery_power_by_time)
    opex_cost_value = cost(consumption_by_time,
                           production_by_time,
                           battery_power_by_time=battery_power_by_time)

    return BasicIndicators(neeg_value, sc_value, ss_value, opex_cost_value)


def calculate_battery_indicators(house: House, pv_plant: PVPlant,
                                 battery: 'Battery'
                                 ) -> BatteryIndicators:
    basic_indicators = calculate_basic_indicators(house, pv_plant, battery)
    battery_protection_value = battery_protection(
        battery.get_battery_soc_by_time())
    battery_variation_value = avg_battery_variation(
        battery.get_battery_soc_by_time(),
    )
    battery_indicators = BatteryIndicators(
        basic_indicators=basic_indicators,
        battery_protection_value=battery_protection_value,
        battery_variation_value=battery_variation_value)

    return battery_indicators


def calculate_indicators_for_cost_triplet(
        house: House,
        min_objective: str,
        pv_plant: PVPlant,
        initial_cost_per_kW: float,
        cost_per_kW: float,
        feed_in_price_per_kW: float,
        battery: Optional['Battery'] = None) -> dict[str, float]:
    """
    Calculate core indicators for a single cost triplet.

    The method computes:
        - NEEG (net energy exchanged with the grid)
        - NPV (net present value)
        - CAPEX (capital expenditure)
        - ROI (return on investment)
        - SC (self-consumption)
        - SS (self-sufficiency)

    Args:
        house: House object with hourly consumption data.
        min_objective: Minimisation objective for the indicator function.
        pv_plant: PVPlant object with hourly production data.
        initial_cost_per_kW: Initial investment cost per kW.
        cost_per_kW: Cost per kWh of electricity.
        feed_in_price_per_kW: Feed-in tariff per kWh.
        battery: Optional battery object, if present.

    Returns:
        Dictionary with keys:
            'neeg', 'npv', 'capex', 'roi', 'sc', 'ss',
            'annual_production_kwh', 'annual_consumption_kwh'.
    """
    if not house.consumption.usage_hourly:
        raise ValueError("Consumption is not set")
    if not pv_plant.production.usage_hourly:
        raise ValueError("Production is not set")

    battery_power_by_time = (
        battery.get_battery_power_by_time() if battery else {})
    consumption_by_time = house.consumption.usage_hourly
    production_by_time = pv_plant.production.usage_hourly

    neeg_value = neeg(
        consumption_by_time,
        production_by_time,
        battery_power_by_time=battery_power_by_time)
    sc_value = self_consumption(
        consumption_by_time,
        production_by_time,
        battery_power_by_time=battery_power_by_time)
    ss_value = self_sufficiency(
        consumption_by_time,
        production_by_time,
        battery_power_by_time=battery_power_by_time)

    annual_production_kwh = sum(production_by_time.values())

    npv_config = NPVConfig(
        initial_cost_per_kW=initial_cost_per_kW,
        cost_per_kW=cost_per_kW,
        feed_in_price_per_kW=feed_in_price_per_kW)

    npv_value, capex, _, _ = npv(
        npv_config,
        pv_plant.config,
        annual_production_kwh,
        sc_value)
    roi_value = npv_value / capex if capex != 0 else 0.0

    annual_consumption_kwh = sum(consumption_by_time.values())

    return {
        "house_id": house.house_id,
        "min_objective": min_objective,
        "neeg": neeg_value,
        "npv": npv_value,
        "capex": capex,
        "roi": roi_value,
        "sc": sc_value,
        "ss": ss_value,
        "annual_production_kwh": annual_production_kwh,
        "annual_consumption_kwh": annual_consumption_kwh,
    }


def export_triplet_indicators_to_csv(
        indicators_list: list[dict[str, float]],
        csv_path: str) -> None:
    """
    Export a list of triplet-indicator dictionaries to a CSV file.

    Each element in ``indicators_list`` should be a dictionary returned by
    :func:`calculate_indicators_for_cost_triplet`. Each dictionary becomes
    one row in the CSV.

    Args:
        indicators_list: List of indicator dictionaries.
        csv_path: Path of the CSV file to write.
    """
    if not indicators_list:
        raise ValueError("indicators_list cannot be empty")

    fieldnames = sorted(indicators_list[0].keys())

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for indicators in indicators_list:
            writer.writerow(indicators)


class Printer:
    def __init__(self, indicators: BasicIndicators):
        self.indicators = indicators

    def print(self, prefix: str = ""):
        """
        Print the indicators with a prefix
        to indicate the source of the indicators.

        Args:
            prefix: Prefix to print before the indicators
        """
        print(f"{prefix}NEEG: {self.indicators.neeg_value:.3f}")
        print(f"{prefix}SC: {self.indicators.sc_value:.3f}")
        print(f"{prefix}SS: {self.indicators.ss_value:.3f}")
        print(f"{prefix}OPEX Cost: {self.indicators.opex_cost_value:.3f}")


class BatteryPrinter:
    def __init__(self, indicators: BatteryIndicators):
        self.indicators = indicators
        self._printer = Printer(indicators.basic_indicators)

    def print(self, prefix: str = ""):
        """
        Print the indicators with a prefix
        to indicate the source of the indicators.
        """
        self._printer.print(prefix)
        bpi_value = self.indicators.battery_protection_value
        avg_variation_value = self.indicators.battery_variation_value
        print(f"{prefix}Battery Protection: {bpi_value:.3f}")
        print(f"{prefix}Battery Variation: {avg_variation_value:.3f}")
