"""
Constants and enumerations used throughout the reno package.

This module defines various constants and enumerations used for:
- Appliance types and categories
- Time and date formats
- Timezone configurations
"""

from enum import Enum


class APPLIANCES(Enum):
    """
    Enumeration of supported household appliances.

    This enumeration defines the types of appliances that can be modeled
    in the energy consumption calculations.

    Attributes:
        TV: Television
        AIR_CONDITIONER: Air conditioning unit
        DISH_WASHER: Dishwashing machine
        ELECTRIC_OVEN: Electric oven
        MICROWAVE: Microwave oven
        WASHING_MACHINE: Washing machine
        CLOTHES_DRYER: Clothes dryer
        FRIDGE: Refrigerator
        OTHER: Other electrical appliances
    """
    TV = "TV"
    AIR_CONDITIONER = "Air Conditioner"
    DISH_WASHER = "Dish Washer"
    ELECTRIC_OVEN = "Electric Oven"
    MICROWAVE = "Microwave"
    WASHING_MACHINE = "Washing Machine"
    CLOTHES_DRYER = "Clothes Dryer"
    FRIDGE = "Fridge"
    CHEST_FREEZER = "Chest Freezer"
    NON_HALOGEN_LAMP_1 = "Non halogen lamp 1"
    NON_HALOGEN_LAMP_2 = "Non halogen lamp 2"
    LIGHT_CONSUMPTION = "Total site light consumption ()"
    OTHER = "Other"


# Timezone names for supported locations
TZ_FRANCE_NAME = 'Europe/Paris'  # Timezone for France
TZ_ROMANIA_NAME = 'Europe/Bucharest'  # Timezone for Romania


location_to_city = {
    "CHATTE": "Valence",
    "GRENOBLE-ST-GEOIRS": "Grenoble",
    "VILLARD-DE-LANS": "Grenoble",
    "CHAMBERY-AIX": "Chambéry",
    "CELLIEU": "Saint-Étienne",
    "MERCUROL": "Valence",
    "BARCELONNETTE": "Barcelonnette",  # small town; alternatively Gap
    "MONTELIMAR": "Montélimar",
    "ST-ETIENNE---MET": "Saint-Étienne",
    "ST-ETIENNE---CT": "Saint-Étienne",
    "ST-ETIENN---MET": "Saint-Étienne",
    "ALBERTVILLE-JO": "Albertville",
    "AMBERIEU": "Ambérieu-en-Bugey",  # near Lyon
    "ALBERTVILLE-J": "Albertville",
    "RIORGES": "Roanne",
    "PERREUX": "Roanne",
    "FIRMINY": "Saint-Étienne",
    "NANTES-BOUGUENAIS": "Nantes",
    "LANDES-GENUSSON": "Cholet",  # or Nantes region
    "GUERANDE": "Saint-Nazaire",
    "ST-MARCEL-LES-V": "Saint-Nazaire",
    "BEAUCOUZE": "Angers",  # nearby, but often grouped with Nantes area
    "ST-NAZAIRE-MONTOIR": "Saint-Nazaire",
    "PTE-DE-CHEMOULIN": "Saint-Nazaire",
    "ST-MARCEL-LES-VAL": "Saint-Nazaire",
    "NIMES-COURBESSAC": "Nîmes",
    "ST-JULIEN-PEYR": "Nîmes",  # or Alès
    "PUJAUT": "Nîmes",
    "DEAUX": "Nîmes",
    "NIMES-GARONS": "Nîmes",
    "TOUR-EIFFEL": "Paris",
    "PARIS-MONTSOURIS": "Paris",
    "VILLACOUBLAY": "Paris",
    "TOUSSUS-LE-NOBLE": "Paris",
    "LA-ROCHE-SUR-YON": "La Roche-sur-Yon",
    "FAGNIERES": "Châlons-en-Champagne",
    "REIMS-COURCY": "Reims",
    "CHAMBRECY": "Reims",
    "VATRY-AERO": "Châlons-en-Champagne",
    # Variations / typos that appear frequently
    "ST-ETIENNN": "Saint-Étienne",
    "ST-ETIENNE": "Saint-Étienne",
    "TOUR-EIFFEL": "Paris",  # already covered
    "PARIS-MONTSOURIST": "Paris",  # typo for Montsouris
    "VILLACOUBL": "Paris",
    "FAGNIER": "Châlons-en-Champagne",
}

MONTH_TO_SEASON = {
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
    12: "winter",
}

ACTIONABLE_BY_PRESENCE = (APPLIANCES.TV,
                          APPLIANCES.AIR_CONDITIONER,
                          APPLIANCES.DISH_WASHER,
                          APPLIANCES.ELECTRIC_OVEN,
                          APPLIANCES.MICROWAVE,
                          APPLIANCES.WASHING_MACHINE,
                          APPLIANCES.CLOTHES_DRYER,
                          APPLIANCES.NON_HALOGEN_LAMP_1,
                          APPLIANCES.NON_HALOGEN_LAMP_2)


SHIFTABLE_APPLIANCES = (APPLIANCES.TV,
                        APPLIANCES.AIR_CONDITIONER,
                        APPLIANCES.DISH_WASHER,
                        APPLIANCES.ELECTRIC_OVEN,
                        APPLIANCES.MICROWAVE,
                        APPLIANCES.WASHING_MACHINE,
                        APPLIANCES.CLOTHES_DRYER,
                        APPLIANCES.NON_HALOGEN_LAMP_1,
                        APPLIANCES.NON_HALOGEN_LAMP_2)
