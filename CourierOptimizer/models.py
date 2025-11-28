# models.py
"""
Core data models for the CourierOptimizer package.
Defines custom exceptions, transport modes, delivery stops, and depot.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from math import isclose
from typing import Dict, Iterable, Optional


class InvalidDataError(Exception):
    """Custom exception for data validation errors."""
    def __init__(self, message: str, row_data: Optional[Iterable[str]] = None):
        self.message = message
        self.row_data = row_data
        # Include row_data in args for better default formatting by tooling
        super().__init__(message, row_data)

    def __str__(self) -> str:
        if self.row_data:
            return f"InvalidDataError: {self.message} (Row: {self.row_data})"
        return f"InvalidDataError: {self.message}"


class Priority(Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

    @property
    def multiplier(self) -> float:
        # High urgency -> lower multiplier (e.g., schedule/cost weighting)
        return {
            Priority.HIGH: 0.6,
            Priority.MEDIUM: 1.0,
            Priority.LOW: 1.2,
        }[self]


@dataclass(frozen=True, slots=True)
class TransportMode:
    """Represents a transport mode with its properties."""
    name: str
    speed_kph: float
    cost_per_km: float
    co2_per_km: float  # grams/km

    def __post_init__(self):
        if self.speed_kph <= 0:
            raise ValueError("speed_kph must be > 0")
        if self.cost_per_km < 0:
            raise ValueError("cost_per_km cannot be negative")
        if self.co2_per_km < 0:
            raise ValueError("co2_per_km cannot be negative")

    @classmethod
    def get_modes_config(cls) -> Dict[str, "TransportMode"]:
        """Returns a dictionary of available transport modes."""
        return {
            "Car": cls("Car", speed_kph=50.0, cost_per_km=4.0, co2_per_km=120.0),
            "Bicycle": cls("Bicycle", speed_kph=15.0, cost_per_km=0.0, co2_per_km=0.0),
            "Walking": cls("Walking", speed_kph=5.0, cost_per_km=0.0, co2_per_km=0.0),
        }

    @classmethod
    def get_mode(cls, mode_name: str) -> "TransportMode":
        """Get a specific transport mode by name."""
        modes = cls.get_modes_config()
        try:
            return modes[mode_name]
        except KeyError as e:
            raise ValueError(f"Unknown transport mode: {mode_name}. Available: {list(modes.keys())}") from e


@dataclass(slots=True)
class Delivery:
    """Represents a delivery stop with customer details and location."""
    customer: str
    latitude: float
    longitude: float
    priority: Priority
    weight_kg: float
    # define on base to avoid attribute errors
    is_depot: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if not (-90.0 <= self.latitude <= 90.0):
            raise InvalidDataError("Latitude must be between -90 and 90", row_data=[self.customer, str(self.latitude)])
        if not (-180.0 <= self.longitude <= 180.0):
            raise InvalidDataError("Longitude must be between -180 and 180", row_data=[self.customer, str(self.longitude)])
        if self.weight_kg < 0:
            raise InvalidDataError("weight_kg cannot be negative", row_data=[self.customer, str(self.weight_kg)])

    def get_priority_multiplier(self) -> float:
        return self.priority.multiplier

    def __repr__(self) -> str:
        return (f"Delivery(customer='{self.customer}', lat={self.latitude}, "
                f"lon={self.longitude}, priority='{self.priority.value}', "
                f"weight_kg={self.weight_kg})")

    def __eq__(self, other: object) -> bool:
        """Two deliveries are equal if they have the same customer and near-identical coordinates."""
        if not isinstance(other, Delivery):
            return False
        # tolerant float compare (about 0.1 meter tolerance ~1e-6 degrees)
        return (
            self.customer == other.customer
            and isclose(self.latitude, other.latitude, abs_tol=1e-6)
            and isclose(self.longitude, other.longitude, abs_tol=1e-6)
        )

    def __hash__(self) -> int:
        # hash consistent with __eq__; round coords to tolerance bucket
        return hash((self.customer, round(self.latitude, 6), round(self.longitude, 6)))


class Depot(Delivery):
    """Represents the depot (start/end point), inherits from Delivery."""
    def __init__(self, latitude: float, longitude: float, name: str = "DEPOT"):
        super().__init__(
            customer=name,
            latitude=latitude,
            longitude=longitude,
            priority=Priority.MEDIUM,  # Neutral value that won't skew calculations
            weight_kg=0.0,
        )
        self.is_depot = True  # overrides base False

    def get_priority_multiplier(self) -> float:
        return 1.0  # Neutral value

    def __repr__(self) -> str:
        return f"Depot(name='{self.customer}', lat={self.latitude}, lon={self.longitude})"


# Example usage and testing
if __name__ == "__main__":
    # Test the custom exception
    try:
        raise InvalidDataError("Invalid latitude value", ["Customer1", "91.0", "10.0", "High", "5.0"])
    except InvalidDataError as e:
        print(f"Exception test: {e}")

    # Test transport modes
    modes = TransportMode.get_modes_config()
    print("\nAvailable transport modes:")
    for name, mode in modes.items():
        print(f"  {name}: {mode}")

    # Test individual mode retrieval
    car_mode = TransportMode.get_mode("Car")
    print(f"\nCar mode details: {car_mode}")

    # Test delivery creation
    delivery1 = Delivery("Customer A", 59.9139, 10.7522, Priority.HIGH, 2.5)
    print(f"\nDelivery 1: {delivery1}")
    print(f"Priority multiplier: {delivery1.get_priority_multiplier()}")

    # Test depot creation
    depot = Depot(59.911, 10.750)
    print(f"\nDepot: {depot}")
    print(f"Depot priority multiplier: {depot.get_priority_multiplier()}")