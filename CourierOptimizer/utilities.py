# utilities.py
"""
Utility functions and metaprogramming tools for CourierOptimizer package.
Includes Haversine distance calculation and timing decorator.
"""

from __future__ import annotations

import math
import time
import functools
import logging
import inspect
from typing import Callable, Any, TYPE_CHECKING, TypeVar, ParamSpec, Dict

if TYPE_CHECKING:
    from models import Delivery  # only for type hints

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

class HaversineCalculator:
    """Calculates great-circle distance between two points on Earth using the Haversine formula."""
    EARTH_RADIUS_KM = 6371.0

    @staticmethod
    def calculate(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Return distance in kilometers."""
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon / 2) ** 2)

        # Clamp to [0, 1] to avoid FP rounding issues
        a = min(1.0, max(0.0, a))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return HaversineCalculator.EARTH_RADIUS_KM * c

    @staticmethod
    def calculate_delivery_distance(delivery1: "Delivery", delivery2: "Delivery") -> float:
        """Calculate distance between two Delivery objects (km)."""
        return HaversineCalculator.calculate(
            delivery1.latitude, delivery1.longitude,
            delivery2.latitude, delivery2.longitude
        )

def _safe_serialize_params(args: tuple[Any, ...], kwargs: dict[str, Any], max_len: int = 500) -> dict[str, Any]:
    def _short(x: Any) -> str:
        try:
            s = repr(x)
        except Exception:
            s = f"<unrepr {type(x).__name__}>"
        if len(s) > max_len:
            s = s[:max_len] + "...[truncated]"
        return s
    return {
        "args": tuple(_short(a) for a in args),
        "kwargs": {k: _short(v) for k, v in kwargs.items()},
    }

def timing_decorator(log_file: str = "run.log") -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator that measures and logs function execution time.
    Writes a simple line to `log_file` and logs via the module logger.
    Preserves the wrapped function's type signature.
    """
    # Optional: switch to logging.FileHandler if you want locking/rotation
    lock = functools.lru_cache(maxsize=1)(lambda: None)  # placeholder to keep a shared object

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        is_coro = inspect.iscoroutinefunction(func)
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        first_is_self = params and params[0].name == "self"

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.perf_counter()
            logger.info("Starting %s at %s", func.__name__, time.strftime('%Y-%m-%d %H:%M:%S'))

            # Log function parameters (omit self for methods)
            log_args = args[1:] if first_is_self and len(args) >= 1 else args
            logger.info("Function %s called with: %s", func.__name__, _safe_serialize_params(log_args, kwargs))

            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                logger.info("Function %s completed in %.4f seconds (%.2f ms)", func.__name__, duration, duration*1000)

                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {func.__name__} - {duration:.4f}s - SUCCESS\n")
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                logger.error("Function %s failed after %.4f seconds with error: %s", func.__name__, duration, e)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {func.__name__} - {duration:.4f}s - ERROR: {e}\n")
                raise

        if not is_coro:
            return sync_wrapper  # type: ignore[return-value]

        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.perf_counter()
            logger.info("Starting %s at %s", func.__name__, time.strftime('%Y-%m-%d %H:%M:%S'))

            sig = inspect.signature(func)
            params = list(sig.parameters.values())
            first_is_self = params and params[0].name == "self"
            log_args = args[1:] if first_is_self and len(args) >= 1 else args
            logger.info("Function %s called with: %s", func.__name__, _safe_serialize_params(log_args, kwargs))

            try:
                result = await func(*args, **kwargs)  # type: ignore[misc]
                duration = time.perf_counter() - start_time
                logger.info("Function %s completed in %.4f seconds (%.2f ms)", func.__name__, duration, duration*1000)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {func.__name__} - {duration:.4f}s - SUCCESS\n")
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                logger.error("Function %s failed after %.4f seconds with error: %s", func.__name__, duration, e)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {func.__name__} - {duration:.4f}s - ERROR: {e}\n")
                raise

        return functools.wraps(func)(async_wrapper)  # type: ignore[return-value]

    return decorator

class DynamicConfigLoader:
    """
    Dynamic configuration loading and class modification.
    """

    @staticmethod
    def add_priority_level(cls: type, priority_name: str, multiplier: float) -> None:
        """
        Add or update a priority multiplier on the class at runtime.
        """
        if not hasattr(cls, '_priority_multipliers') or not isinstance(getattr(cls, '_priority_multipliers'), dict):
            setattr(cls, '_priority_multipliers', {"High": 0.6, "Medium": 1.0, "Low": 1.2})

        cls._priority_multipliers[priority_name] = multiplier  # type: ignore[attr-defined]

        # Inject/replace a safe instance method
        def get_priority_multiplier(self) -> float:
            multipliers: Dict[str, float] = getattr(self.__class__, "_priority_multipliers", {})  # type: ignore[attr-defined]
            priority = getattr(self, "priority", None)
            return multipliers.get(priority, 1.0)

        setattr(cls, "get_priority_multiplier", get_priority_multiplier)
        logger.info("Added/updated priority level '%s' with multiplier %s", priority_name, multiplier)

    @staticmethod
    def add_transport_mode(mode_name: str, speed_kph: float, cost_per_km: float, co2_per_km: float) -> None:
        """
        Dynamically add a new transport mode at runtime.
        """
        from models import TransportMode

        modes = dict(TransportMode.get_modes_config())  # copy to avoid external mutation
        modes[mode_name] = TransportMode(
            name=mode_name,
            speed_kph=speed_kph,
            cost_per_km=cost_per_km,
            co2_per_km=co2_per_km,
        )

        @classmethod
        def new_get_modes_config(cls):
            # return a copy to avoid accidental external mutation
            return dict(modes)

        TransportMode.get_modes_config = new_get_modes_config  # type: ignore[assignment]
        logger.info("Added new transport mode '%s'", mode_name)

if __name__ == "__main__":
    # Demo logging only for the test run
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print("Testing Haversine distance calculation:")
    oslo_lat, oslo_lon = 59.9139, 10.7522
    bergen_lat, bergen_lon = 60.3913, 5.3221

    distance = HaversineCalculator.calculate(oslo_lat, oslo_lon, bergen_lat, bergen_lon)
    print(f"Oslo to Bergen: {distance:.2f} km")

    # Local import to avoid circulars at module import time
    from models import Delivery, TransportMode

    delivery1 = Delivery("Test1", oslo_lat, oslo_lon, "High", 2.0)
    delivery2 = Delivery("Test2", bergen_lat, bergen_lon, "Medium", 3.0)
    delivery_distance = HaversineCalculator.calculate_delivery_distance(delivery1, delivery2)
    print(f"Delivery distance: {delivery_distance:.2f} km")

    print("\nTesting timing decorator (sync):")
    @timing_decorator("test_run.log")
    def test_function(seconds: float):
        time.sleep(seconds)
        return f"Slept for {seconds} seconds"

    result = test_function(0.1)
    print(f"Function result: {result}")

    @timing_decorator("test_run.log")
    def failing_function():
        raise ValueError("This is a test error")

    try:
        failing_function()
    except ValueError as e:
        print(f"Expected error caught: {e}")

    print("\nTesting dynamic configuration loading:")
    DynamicConfigLoader.add_priority_level(Delivery, "Urgent", 0.3)
    urgent_delivery = Delivery("Urgent Customer", 59.914, 10.751, "Urgent", 1.0)
    print(f"Urgent priority multiplier: {urgent_delivery.get_priority_multiplier()}")

    DynamicConfigLoader.add_transport_mode("Scooter", 25.0, 1.5, 15.0)
    scooter_mode = TransportMode.get_mode("Scooter")
    print(f"New transport mode: {scooter_mode}")

    print("\nAll utilities tests completed!")