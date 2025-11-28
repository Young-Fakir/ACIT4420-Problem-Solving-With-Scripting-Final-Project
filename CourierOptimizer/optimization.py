# optimization.py
"""
Route optimization module for CourierOptimizer package.
Implements two-stage local search with cost calculation, nearest neighbor, and 2-opt algorithms.
Includes Pareto multi-objective optimization for the optional assignment requirement.
"""

import logging
from typing import List, Tuple, Dict, Any, Union

from models import Delivery, Depot, TransportMode, Priority
from utilities import HaversineCalculator

logger = logging.getLogger(__name__)

# Type alias for mixed stop lists
StopT = Union[Delivery, Depot]


class RouteOptimizer:
    """Main optimization class that implements the two-stage local search algorithm."""

    def __init__(self, depot: Depot, deliveries: List[Delivery],
                 transport_mode: TransportMode, objective: str):
        """
        Initialize the route optimizer.

        Args:
            depot: The start/end depot
            deliveries: List of delivery stops
            transport_mode: Selected transport mode
            objective: Optimization objective ('Time', 'Cost', or 'CO2')
        """
        self.depot = depot
        self.deliveries = deliveries
        self.transport_mode = transport_mode
        self.objective = objective.strip().lower()

        # Validate objective
        valid_objectives = ['time', 'cost', 'co2']
        if self.objective not in valid_objectives:
            raise ValueError(f"Invalid objective: {objective}. Must be one of {valid_objectives}")

        # Validate transport mode speed for time metrics (used in metrics regardless of objective)
        if self.transport_mode.speed_kph <= 0:
            raise ValueError("transport_mode.speed_kph must be > 0")

        # Combined list of all stops (depot + deliveries)
        self.all_stops: List[StopT] = [self.depot] + self.deliveries
        self.stop_count = len(self.all_stops)

        # Fast object-id -> index map to avoid O(n) .index() in hot paths
        self._idx: Dict[int, int] = {id(s): i for i, s in enumerate(self.all_stops)}

        # Pre-calculate distance matrix for performance (symmetric)
        self.distance_matrix = self._precalculate_distances()

        logger.info(
            f"Initialized RouteOptimizer with {len(deliveries)} deliveries, "
            f"mode: {transport_mode.name}, objective: {objective}"
        )

    def _precalculate_distances(self) -> List[List[float]]:
        """Pre-calculate distance matrix between all stops for performance."""
        matrix = [[0.0] * self.stop_count for _ in range(self.stop_count)]

        for i in range(self.stop_count):
            for j in range(i + 1, self.stop_count):
                distance = HaversineCalculator.calculate_delivery_distance(
                    self.all_stops[i], self.all_stops[j]
                )
                matrix[i][j] = distance
                matrix[j][i] = distance

        return matrix

    # ---------- Cost helpers (index-based) ----------

    def _weighted_cost_idx(self, i: int, j: int) -> float:
        """Weighted cost between indexed stops i -> j (destination priority multiplier)."""
        distance = self.distance_matrix[i][j]

        if self.objective == 'cost':
            rate = self.transport_mode.cost_per_km
        elif self.objective == 'co2':
            rate = self.transport_mode.co2_per_km
        else:  # 'time'
            rate = 1.0 / self.transport_mode.speed_kph  # hours per km

        # Safe priority multiplier (Depot may not implement it)
        priority_multiplier = getattr(self.all_stops[j], "get_priority_multiplier", lambda: 1.0)()
        return distance * rate * priority_multiplier

    def _calculate_total_tour_cost_idx(self, tour_idx: List[int]) -> Tuple[float, Dict[str, float]]:
        """Total weighted cost + metrics for a full tour given as indices."""
        total_weighted_cost = 0.0
        total_distance = 0.0
        total_time_hours = 0.0
        total_cost_nok = 0.0
        total_co2_g = 0.0

        for a, b in zip(tour_idx, tour_idx[1:]):
            d = self.distance_matrix[a][b]
            total_distance += d
            total_time_hours += d / self.transport_mode.speed_kph
            total_cost_nok += d * self.transport_mode.cost_per_km
            total_co2_g += d * self.transport_mode.co2_per_km
            total_weighted_cost += self._weighted_cost_idx(a, b)

        metrics = {
            'total_distance_km': total_distance,
            'total_time_hours': total_time_hours,
            'total_cost_nok': total_cost_nok,
            'total_co2_g': total_co2_g,
            'total_weighted_cost': total_weighted_cost
        }
        return total_weighted_cost, metrics

    # ---------- Public cost API (object-based) ----------

    def calculate_weighted_cost(self, from_stop: StopT, to_stop: StopT) -> float:
        """
        Calculate weighted cost between two stops (Step 1 of the algorithm).

        Args:
            from_stop: Starting stop (Depot or Delivery)
            to_stop: Destination stop (Depot or Delivery)

        Returns:
            Weighted cost for this segment
        """
        from_idx = self._idx[id(from_stop)]
        to_idx = self._idx[id(to_stop)]
        return self._weighted_cost_idx(from_idx, to_idx)

    def calculate_total_tour_cost(self, tour: List[StopT]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total cost and breakdown for a complete tour.

        Args:
            tour: Ordered list of stops (Depot/Delivery)

        Returns:
            Tuple of (total_weighted_cost, metrics_dict)
        """
        try:
            tour_idx = [self._idx[id(s)] for s in tour]
        except KeyError as e:
            raise ValueError("Tour contains stops not in optimizer.all_stops") from e
        return self._calculate_total_tour_cost_idx(tour_idx)

    # ---------- Construction: Nearest Neighbor (index-based) ----------

    def generate_initial_tour(self) -> List[StopT]:
        """
        Generate initial tour using Nearest Neighbor algorithm (Step 2).

        Returns:
            Ordered list of stops representing the initial tour
        """
        # Start at depot (index 0)
        tour_idx: List[int] = [0]
        unvisited: List[int] = list(range(1, self.stop_count))  # delivery indices

        while unvisited:
            i = tour_idx[-1]
            # deterministic tie-break via index
            best_j = min(unvisited, key=lambda j: (self._weighted_cost_idx(i, j), j))
            tour_idx.append(best_j)
            unvisited.remove(best_j)

        # Return to depot
        tour_idx.append(0)
        tour: List[StopT] = [self.all_stops[k] for k in tour_idx]

        logger.info(f"Generated initial tour with {len(tour)} stops")
        return tour

    # ---------- 2-opt local search (index-based) ----------

    @staticmethod
    def _perform_2opt_swap_idx(tour_idx: List[int], i: int, j: int) -> List[int]:
        """
        Perform a 2-opt swap on the index tour between i and j (inclusive).
        T_new = T[0..i-1] + reverse(T[i..j]) + T[j+1..end]
        """
        return tour_idx[:i] + list(reversed(tour_idx[i:j + 1])) + tour_idx[j + 1:]

    def optimize_2_opt(self, initial_tour: List[StopT], max_iterations: int = 1000) -> List[StopT]:
        """
        Improve tour using 2-opt local search (Step 3).

        Args:
            initial_tour: Starting tour from nearest neighbor
            max_iterations: Maximum number of iterations to prevent infinite loops

        Returns:
            Optimized tour (list of Delivery/Depot objects)
        """
        # Short-circuit for tiny tours (depot->depot or single delivery loops)
        if len(initial_tour) <= 3:
            return initial_tour

        # Work on indices for speed
        current_tour_idx = [self._idx[id(s)] for s in initial_tour]
        start_cost, _ = self._calculate_total_tour_cost_idx(current_tour_idx)
        current_cost = start_cost

        improved = True
        iteration = 0

        logger.info(f"Starting 2-opt optimization with initial cost: {start_cost:.4f}")

        n = len(current_tour_idx)
        # Keep depot fixed at start/end; iterate over inner edges
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for i in range(1, n - 2):           # skip depot at index 0
                for j in range(i + 1, n - 1):   # keep last depot fixed
                    new_tour_idx = self._perform_2opt_swap_idx(current_tour_idx, i, j)
                    new_cost, _ = self._calculate_total_tour_cost_idx(new_tour_idx)

                    if new_cost < current_cost - 1e-12:  # small epsilon to avoid float churn
                        current_tour_idx = new_tour_idx
                        current_cost = new_cost
                        improved = True
                        logger.debug(f"Iteration {iteration}: Improved cost to {current_cost:.6f} (swap {i}-{j})")
                        break  # restart search from beginning after an improvement
                if improved:
                    break

        final_cost, _ = self._calculate_total_tour_cost_idx(current_tour_idx)
        improvement = ((start_cost - final_cost) / start_cost * 100.0) if start_cost > 0 else 0.0

        logger.info(
            f"2-opt completed after {iteration} iterations. "
            f"Final cost: {final_cost:.4f} ({improvement:.2f}% improvement)"
        )

        return [self.all_stops[k] for k in current_tour_idx]

    # ---------- Orchestration ----------

    def optimize_route(self) -> Tuple[List[StopT], Dict[str, Any]]:
        """
        Main optimization method that runs the complete two-stage algorithm.

        Returns:
            Tuple of (optimized_tour, optimization_results)
        """
        logger.info("Starting two-stage route optimization")

        # Stage 1: Generate initial tour with Nearest Neighbor
        initial_tour = self.generate_initial_tour()
        initial_cost, initial_metrics = self.calculate_total_tour_cost(initial_tour)

        # Stage 2: Improve tour with 2-opt
        optimized_tour = self.optimize_2_opt(initial_tour)
        final_cost, final_metrics = self.calculate_total_tour_cost(optimized_tour)

        # Calculate improvement
        improvement_pct = ((initial_cost - final_cost) / initial_cost * 100.0) if initial_cost > 0 else 0.0

        results = {
            'initial_tour': initial_tour,
            'optimized_tour': optimized_tour,
            'initial_cost': initial_cost,
            'final_cost': final_cost,
            'improvement_pct': improvement_pct,
            'initial_metrics': initial_metrics,
            'final_metrics': final_metrics,
            'transport_mode': self.transport_mode.name,
            'objective': self.objective
        }

        logger.info(f"Optimization complete: {improvement_pct:.2f}% improvement")

        return optimized_tour, results


class WeightedRouteOptimizer(RouteOptimizer):
    """Route optimizer with configurable weights for multi-objective optimization."""

    def __init__(self, depot: Depot, deliveries: List[Delivery],
                 transport_mode: TransportMode, weights: Dict[str, float]):
        """
        Initialize weighted route optimizer.

        Args:
            depot: The start/end depot
            deliveries: List of delivery stops
            transport_mode: Selected transport mode
            weights: Dictionary with time_weight, cost_weight, co2_weight
        """
        # Initialize with 'time' objective (we override cost calculation anyway)
        super().__init__(depot, deliveries, transport_mode, "time")

        required = ("time_weight", "cost_weight", "co2_weight")
        missing = [k for k in required if k not in weights]
        if missing:
            raise ValueError(f"Missing weight keys: {missing}")
        if any(weights[k] < 0 for k in required):
            raise ValueError("Weights must be non-negative")

        total = sum(weights[k] for k in required)
        if abs(total - 1.0) > 0.001:
            logger.warning(f"Weights sum to {total:.3f}, should be ≈1.0 for proper scaling")

        self.weights = weights

    def _weighted_cost_idx(self, i: int, j: int) -> float:
        """Calculate cost using configurable weights for multi-objective optimization."""
        distance = self.distance_matrix[i][j]

        # Individual objective contributions
        time_cost = distance / self.transport_mode.speed_kph
        cost_component = distance * self.transport_mode.cost_per_km
        co2_component = distance * self.transport_mode.co2_per_km

        # Apply weights to combine objectives
        weighted_cost = (
            self.weights['time_weight'] * time_cost +
            self.weights['cost_weight'] * cost_component +
            self.weights['co2_weight'] * co2_component
        )

        # Apply priority multiplier safely
        priority_multiplier = getattr(self.all_stops[j], "get_priority_multiplier", lambda: 1.0)()
        return weighted_cost * priority_multiplier


class ParetoOptimizer:
    """Generates Pareto-optimal solutions for multi-objective optimization."""

    def __init__(self, depot: Depot, deliveries: List[Delivery], transport_mode: TransportMode):
        """
        Initialize Pareto optimizer.

        Args:
            depot: The start/end depot
            deliveries: List of delivery stops
            transport_mode: Selected transport mode
        """
        self.depot = depot
        self.deliveries = deliveries
        self.transport_mode = transport_mode

        logger.info("Initialized ParetoOptimizer for multi-objective optimization")

    def generate_pareto_front(self, num_solutions: int = 8) -> List[Dict[str, Any]]:
        """
        Generate Pareto-optimal solutions using different weight combinations.

        Args:
            num_solutions: Number of Pareto solutions to generate

        Returns:
            List of non-dominated solutions with their metrics
        """
        logger.info(f"Generating Pareto front with {num_solutions} solutions")

        # Generate different weight combinations
        weight_combinations = self._generate_weight_combinations(num_solutions)

        solutions: List[Dict[str, Any]] = []

        for i, weights in enumerate(weight_combinations):
            logger.info(f"Evaluating solution {i+1}/{num_solutions} with weights {weights}")

            try:
                # Create optimizer with specific weights
                optimizer = WeightedRouteOptimizer(
                    self.depot, self.deliveries, self.transport_mode, weights
                )

                # Optimize route with these weights
                tour, results = optimizer.optimize_route()

                solution = {
                    'solution_id': i + 1,
                    'weights': weights.copy(),  # Store a copy
                    'tour': tour,
                    'time_hours': results['final_metrics']['total_time_hours'],
                    'cost_nok': results['final_metrics']['total_cost_nok'],
                    'co2_g': results['final_metrics']['total_co2_g'],
                    'distance_km': results['final_metrics']['total_distance_km'],
                    'improvement_pct': results['improvement_pct']
                }

                solutions.append(solution)

            except Exception as e:
                logger.error(f"Failed to evaluate solution with weights {weights}: {e}")
                continue

        # Filter to non-dominated solutions (Pareto front)
        pareto_solutions = self._filter_pareto_front(solutions)
        logger.info(f"Pareto front contains {len(pareto_solutions)} non-dominated solutions")

        return pareto_solutions

    def _generate_weight_combinations(self, num_combinations: int) -> List[Dict[str, float]]:
        """Generate different weight combinations for time, cost, and CO2."""
        combinations: List[Dict[str, float]] = []

        if num_combinations == 1:
            # Single balanced combination
            return [{'time_weight': 0.34, 'cost_weight': 0.33, 'co2_weight': 0.33}]

        # Generate weights that sum to 1.0
        for i in range(num_combinations):
            # Vary time weight from 0 to 1
            time_weight = i / (num_combinations - 1) if num_combinations > 1 else 0.5

            # Distribute remaining weight between cost and CO2
            remaining = 1.0 - time_weight
            cost_weight = remaining * 0.5  # Half to cost
            co2_weight = remaining * 0.5   # Half to CO2

            combinations.append({
                'time_weight': round(time_weight, 3),
                'cost_weight': round(cost_weight, 3),
                'co2_weight': round(co2_weight, 3)
            })

        return combinations

    def _filter_pareto_front(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter solutions to only include non-dominated ones (Pareto front)."""
        if not solutions:
            return []

        pareto_front: List[Dict[str, Any]] = []

        for solution in solutions:
            dominated = False

            # Check if this solution is dominated by any other
            for other in solutions:
                if solution is other:
                    continue

                # A solution is dominated if another is better in ALL objectives
                # (lower values are better for all objectives)
                if (other['time_hours'] <= solution['time_hours'] and
                    other['cost_nok'] <= solution['cost_nok'] and
                    other['co2_g'] <= solution['co2_g'] and
                    (other['time_hours'] < solution['time_hours'] or
                     other['cost_nok'] < solution['cost_nok'] or
                     other['co2_g'] < solution['co2_g'])):
                    dominated = True
                    break

            if not dominated:
                pareto_front.append(solution)

        # Sort by time for consistent display
        pareto_front.sort(key=lambda x: x['time_hours'])

        return pareto_front


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("Testing RouteOptimizer...")

    try:
        # NOTE: Import Priority for correct Delivery construction
        from models import Priority  # <-- added

        # Create test data
        depot = Depot(59.911, 10.750)

        # Use Priority enum values instead of strings
        deliveries = [
            Delivery("Customer A", 59.9139, 10.7522, Priority.HIGH,   2.5),
            Delivery("Customer B", 59.9150, 10.7490, Priority.MEDIUM, 1.5),
            Delivery("Customer C", 59.9100, 10.7550, Priority.LOW,    3.0),
            Delivery("Customer D", 59.9120, 10.7480, Priority.HIGH,   1.0),
        ]

        # Get transport mode
        transport_mode = TransportMode.get_mode("Car")

        # Test with different objectives
        objectives = ['Time', 'Cost', 'CO2']

        for objective in objectives:
            print(f"\n{'='*50}")
            print(f"Testing optimization with objective: {objective}")
            print(f"{'='*50}")

            optimizer = RouteOptimizer(depot, deliveries, transport_mode, objective)
            optimized_tour, results = optimizer.optimize_route()

            print(f"Initial cost: {results['initial_cost']:.4f}")
            print(f"Final  cost: {results['final_cost']:.4f}")
            print(f"Improvement: {results['improvement_pct']:.2f}%")

            print(f"\nOptimized tour ({len(optimized_tour)} stops):")
            for i, stop in enumerate(optimized_tour):
                is_depot = getattr(stop, "is_depot", False)
                stop_type = "Depot" if is_depot else "Delivery"
                name = getattr(stop, "customer", getattr(stop, "name", "Depot" if is_depot else "Delivery"))
                print(f"  {i+1:2d}. {stop_type:8} - {name}")

            print(f"\nFinal metrics:")
            metrics = results['final_metrics']
            print(f"  Total distance: {metrics['total_distance_km']:.2f} km")
            print(f"  Total time:     {metrics['total_time_hours']:.2f} hours "
                  f"({metrics['total_time_hours']*60:.1f} minutes)")
            print(f"  Total cost:     {metrics['total_cost_nok']:.2f} NOK")
            print(f"  Total CO2:      {metrics['total_co2_g']:.1f} g")

        # Test Pareto optimization
        print(f"\n{'='*50}")
        print("Testing Pareto Multi-Objective Optimization")
        print(f"{'='*50}")

        pareto_optimizer = ParetoOptimizer(depot, deliveries, transport_mode)
        pareto_solutions = pareto_optimizer.generate_pareto_front(num_solutions=6)

        print(f"Found {len(pareto_solutions)} Pareto-optimal solutions:")
        print("\nSolution | Time (hours) | Cost (NOK) | CO2 (g) | Weights (T/C/E)")
        print("-" * 65)

        for solution in pareto_solutions:
            weights = solution['weights']
            print(f"{solution['solution_id']:8} | {solution['time_hours']:12.3f} | {solution['cost_nok']:10.2f} | "
                  f"{solution['co2_g']:7.1f} | {weights['time_weight']:.2f}/{weights['cost_weight']:.2f}/"
                  f"{weights['co2_weight']:.2f}")

        print(f"\n{'='*50}")
        print("All optimization tests completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()