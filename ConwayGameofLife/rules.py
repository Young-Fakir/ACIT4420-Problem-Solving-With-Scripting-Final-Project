"""
rules.py - Core Evolution Algorithm for Conway's Game of Life

This module implements the cellular automaton evolution rules and provides
a flexible rule system with metaprogramming capabilities for dynamic extension.
"""

from typing import Callable, Dict, List, Optional, Any
import logging
from functools import wraps
import numpy as np

# Module logger (consistent with board.py)
_logger = logging.getLogger(__name__)

# Custom exceptions
class RuleError(Exception):
    """Raised when rule application fails"""
    pass

class NeighborhoodError(Exception):
    """Raised when neighborhood calculation fails"""
    pass


# Rule registry for dynamic rule loading
RULE_REGISTRY: Dict[str, Callable[[int, int], int]] = {}


def register_rule(rule_name: str) -> Callable:
    """
    Decorator for registering new rule sets dynamically.

    Args:
        rule_name: Unique identifier for the rule set

    Returns:
        Decorator function
    """
    def decorator(func: Callable[[int, int], int]) -> Callable[[int, int], int]:
        if rule_name in RULE_REGISTRY:
            _logger.warning("Overwriting existing rule: %s", rule_name)
        RULE_REGISTRY[rule_name] = func
        _logger.info("Registered rule set: %s", rule_name)
        return func
    return decorator


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to log execution time of evolution functions.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        from time import perf_counter
        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed = perf_counter() - start
        _logger.info("%s executed in %.4f seconds", func.__name__, elapsed)
        return result
    return wrapper


class Neighborhood:
    """
    Handles different neighborhood types for cellular automata.
    """

    @staticmethod
    def moore_neighborhood(
        grid: np.ndarray,
        x: int,
        y: int,
        boundary: str = 'fixed'
    ) -> int:
        """
        Calculate Moore neighborhood (8 surrounding cells).

        Args:
            grid: 2D array of cell states (0/1 or bool)
            x: X coordinate of center cell
            y: Y coordinate of center cell
            boundary: 'fixed' (dead outside) or 'toroidal' (wrapping)

        Returns:
            Count of live neighbors

        Raises:
            NeighborhoodError: If coordinates are invalid or boundary is unknown
        """
        if boundary not in ('fixed', 'toroidal'):
            raise NeighborhoodError(f"Unknown boundary mode: {boundary}")

        height, width = grid.shape
        live_count: int = 0

        try:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue  # Skip center

                    nx, ny = x + dx, y + dy

                    if boundary == 'toroidal':
                        nx %= width
                        ny %= height
                        live_count += int(grid[ny, nx])
                    else:
                        if 0 <= nx < width and 0 <= ny < height:
                            live_count += int(grid[ny, nx])

            return live_count

        except IndexError as e:
            raise NeighborhoodError(
                f"Invalid coordinates ({x}, {y}) for grid shape {grid.shape}"
            ) from e


@register_rule("conway")
def conway_rules(current_state: int, live_neighbors: int) -> int:
    """
    Conway's original Game of Life rules.

    Rules:
    1. Underpopulation: Live cell with <2 neighbors dies
    2. Survival: Live cell with 2-3 neighbors lives
    3. Overpopulation: Live cell with >3 neighbors dies
    4. Reproduction: Dead cell with exactly 3 neighbors becomes alive

    Args:
        current_state: Current cell state (0=dead, 1=alive)
        live_neighbors: Number of live neighbors

    Returns:
        Next state of the cell (0=dead, 1=alive)
    """
    if current_state == 1:
        return 1 if live_neighbors in (2, 3) else 0
    else:
        return 1 if live_neighbors == 3 else 0


@register_rule("high_life")
def high_life_rules(current_state: int, live_neighbors: int) -> int:
    """
    HighLife variant (B36/S23) - similar to Conway but with extra reproduction rule.

    - Birth: 3 or 6 neighbors
    - Survival: 2 or 3 neighbors
    """
    if current_state == 1:
        return 1 if live_neighbors in (2, 3) else 0
    else:
        return 1 if live_neighbors in (3, 6) else 0


@register_rule("day_and_night")
def day_and_night_rules(current_state: int, live_neighbors: int) -> int:
    """
    Day & Night (B3678/S34678) - symmetric rule set.
    """
    if current_state == 1:
        return 1 if live_neighbors in (3, 4, 6, 7, 8) else 0
    else:
        return 1 if live_neighbors in (3, 6, 7, 8) else 0


@register_rule("life_34")
def life_34_rules(current_state: int, live_neighbors: int) -> int:
    """
    34 Life (B34/S34) - cells survive and are born with 3 or 4 neighbors.
    """
    return 1 if live_neighbors in (3, 4) else 0


class RuleEngine:
    """
    Engine for applying cellular automata rules to grids.
    """

    def __init__(self, rule_name: str = "conway", boundary: str = "fixed"):
        """
        Initialize rule engine with specified rule set and boundary conditions.

        Args:
            rule_name: Name of rule set to use
            boundary: 'fixed' or 'toroidal'

        Raises:
            RuleError: If rule name is not found in registry
        """
        self.rule_name = rule_name
        self.boundary = boundary
        self.rule_function = self._get_rule_function(rule_name)
        self.neighborhood_calculator = Neighborhood.moore_neighborhood

        if boundary not in ('fixed', 'toroidal'):
            raise ValueError("Boundary must be 'fixed' or 'toroidal'")

        _logger.info("Initialized RuleEngine with rules: %s, boundary: %s",
                     rule_name, boundary)

    def _get_rule_function(self, rule_name: str) -> Callable[[int, int], int]:
        """Get rule function from registry"""
        if rule_name not in RULE_REGISTRY:
            available_rules = list(RULE_REGISTRY.keys())
            raise RuleError(
                f"Rule '{rule_name}' not found. Available rules: {available_rules}"
            )
        return RULE_REGISTRY[rule_name]

    def set_rule(self, rule_name: str) -> None:
        """
        Dynamically change the rule set.

        Raises:
            RuleError: If rule name is not found
        """
        self.rule_function = self._get_rule_function(rule_name)
        self.rule_name = rule_name
        _logger.info("Rule set changed to: %s", rule_name)

    def set_boundary(self, boundary: str) -> None:
        """
        Set boundary condition: 'fixed' or 'toroidal'.
        """
        if boundary not in ('fixed', 'toroidal'):
            raise ValueError("Boundary must be 'fixed' or 'toroidal'")
        self.boundary = boundary
        _logger.info("Boundary condition set to: %s", boundary)

    @timing_decorator
    def evolve_grid(self, current_grid: np.ndarray) -> np.ndarray:
        """
        Apply evolution rules to create next generation grid.

        Notes:
            - Preserves the dtype of `current_grid` (bool recommended).
        """
        try:
            height, width = current_grid.shape
            # Preserve dtype (bool in your board)
            new_grid = np.zeros((height, width), dtype=current_grid.dtype)

            # Operate on ints for counting but keep original values for rules
            for y in range(height):
                for x in range(width):
                    live_neighbors = self.neighborhood_calculator(
                        current_grid, x, y, self.boundary
                    )
                    current_state = int(current_grid[y, x])
                    next_state = self.rule_function(current_state, live_neighbors)
                    # Cast back to target dtype (bool works perfectly)
                    new_grid[y, x] = np.asarray(next_state, dtype=new_grid.dtype)

            return new_grid

        except Exception as e:
            raise RuleError(f"Grid evolution failed: {e}") from e

    def get_rule_info(self) -> Dict[str, Any]:
        """
        Get information about the current rule set.
        """
        return {
            'rule_name': self.rule_name,
            'boundary': self.boundary,
            'rule_function': self.rule_function.__name__,
            'available_rules': list(RULE_REGISTRY.keys())
        }


# Convenience function for easy evolution
@timing_decorator
def evolve_grid(current_grid: np.ndarray,
                rule_name: str = "conway",
                boundary: str = "fixed") -> np.ndarray:
    """
    Convenience function to evolve grid in one call.
    """
    engine = RuleEngine(rule_name, boundary)
    return engine.evolve_grid(current_grid)


def list_available_rules() -> List[str]:
    """Get list of all available rule sets."""
    return list(RULE_REGISTRY.keys())


def add_custom_rule(rule_name: str, rule_function: Callable[[int, int], int]) -> None:
    """
    Add a custom rule function to the registry.

    Raises:
        ValueError: If rule name already exists
    """
    if rule_name in RULE_REGISTRY:
        raise ValueError(f"Rule '{rule_name}' already exists")

    RULE_REGISTRY[rule_name] = rule_function
    _logger.info("Custom rule added: %s", rule_name)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        print("Available rules:", list_available_rules())

        # Create a test grid (blinker pattern), use bool dtype to match board.py
        test_grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=bool)

        print("Initial grid:")
        print(test_grid.astype(int))

        # Test Conway rules
        engine = RuleEngine("conway", boundary="fixed")
        next_gen = engine.evolve_grid(test_grid)

        print("\nNext generation (Conway rules):")
        print(next_gen.astype(int))

        # Test different rules
        high_life_gen = evolve_grid(test_grid, "high_life")
        print("\nNext generation (HighLife rules):")
        print(high_life_gen.astype(int))

        # Test rule info
        info = engine.get_rule_info()
        print(f"\nRule engine info: {info}")

    except Exception as e:
        print(f"Error during testing: {e}")