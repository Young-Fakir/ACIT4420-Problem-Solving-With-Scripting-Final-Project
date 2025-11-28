# main.py
"""
Main entry point for CourierOptimizer package.
Orchestrates the entire routing optimization process with both CLI and text menu interfaces.
"""

import sys
import os
import argparse
import logging
from typing import Dict, Any, List, Tuple, Union

# Import custom modules
from data_validator import DataValidator
from models import Depot, TransportMode, InvalidDataError
from utilities import HaversineCalculator, timing_decorator
from optimization import RouteOptimizer, ParetoOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('run.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    'input_file': 'input_data.csv',
    'depot_lat': 59.911,
    'depot_lon': 10.750,
    'default_mode': 'Car',
    'default_objective': 'Time'
}


class CourierOptimizerInterface:
    """Handles both CLI and text menu interfaces for CourierOptimizer."""

    def __init__(self):
        modes_cfg = TransportMode.get_modes_config()
        # tolerate dict or object values
        self.available_modes = list(modes_cfg.keys())
        self.available_objectives = ['Time', 'Cost', 'CO2', 'Pareto']

    def parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description='Courier Route Optimizer - Find optimal delivery routes',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=f'''
Examples:
  python {sys.argv[0]}                                # Use text menu interface
  python {sys.argv[0]} -i deliveries.csv              # Custom input file
  python {sys.argv[0]} -m Bicycle -o Cost            # Bike for lowest cost
  python {sys.argv[0]} -m Car -o Pareto              # Multi-objective Pareto optimization
  python {sys.argv[0]} --depot 59.912 10.749         # Custom depot location

Available transport modes: {', '.join(self.available_modes)}
Available objectives: {', '.join(self.available_objectives)}
            '''
        )

        parser.add_argument(
            '-i', '--input',
            dest='input_file',
            default=DEFAULT_CONFIG['input_file'],
            help=f'Input CSV file (default: {DEFAULT_CONFIG["input_file"]})'
        )

        parser.add_argument(
            '-m', '--mode',
            choices=self.available_modes,
            help='Transport mode (default: use text menu)'
        )

        parser.add_argument(
            '-o', '--objective',
            choices=self.available_objectives,
            help='Optimization objective (default: use text menu)'
        )

        parser.add_argument(
            '--depot',
            nargs=2,
            type=float,
            metavar=('LAT', 'LON'),
            default=[DEFAULT_CONFIG['depot_lat'], DEFAULT_CONFIG['depot_lon']],
            help=f'Depot coordinates (default: {DEFAULT_CONFIG["depot_lat"]}, {DEFAULT_CONFIG["depot_lon"]})'
        )

        parser.add_argument(
            '--list-modes',
            action='store_true',
            help='List available transport modes and exit'
        )

        parser.add_argument(
            '--text-menu',
            action='store_true',
            help='Force text menu interface (default behavior)'
        )

        parser.add_argument(
            '--version',
            action='store_true',
            help='Show program version and exit'
        )

        return parser.parse_args()

    def display_welcome(self):
        """Display welcome message and program information."""
        print("\n" + "=" * 60)
        print("        COURIER OPTIMIZER - NordicExpress Delivery")
        print("=" * 60)
        print("Optimizing delivery routes for efficiency and sustainability")
        print("=" * 60)

    def display_transport_menu(self) -> str:
        """Display dynamic transport mode selection menu."""
        print("\n" + "=" * 40)
        print("        SELECT TRANSPORT MODE")
        print("=" * 40)
        for idx, name in enumerate(self.available_modes, 1):
            print(f"{idx}. {name}")
        print("=" * 40)

        while True:
            try:
                choice = input(f"Enter choice (1-{len(self.available_modes)}): ").strip()
                if choice.isdigit():
                    idx = int(choice)
                    if 1 <= idx <= len(self.available_modes):
                        return self.available_modes[idx - 1]
                print(f"Invalid choice. Please enter a number between 1 and {len(self.available_modes)}.")
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled.")
                sys.exit(0)

    def display_objective_menu(self) -> str:
        """Display optimization objective selection menu."""
        print("\n" + "=" * 40)
        print("     SELECT OPTIMIZATION OBJECTIVE")
        print("=" * 40)
        for idx, name in enumerate(self.available_objectives, 1):
            label = {
                'Time': 'Fastest Time',
                'Cost': 'Lowest Cost',
                'CO2': 'Lowest CO2 Emissions',
                'Pareto': 'Multi-Objective (Pareto View)'
            }.get(name, name)
            print(f"{idx}. {label}")
        print("=" * 40)

        while True:
            try:
                choice = input(f"Enter choice (1-{len(self.available_objectives)}): ").strip()
                if choice.isdigit():
                    idx = int(choice)
                    if 1 <= idx <= len(self.available_objectives):
                        return self.available_objectives[idx - 1]
                print(f"Invalid choice. Please enter a number between 1 and {len(self.available_objectives)}.")
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled.")
                sys.exit(0)

    def get_depot_coordinates(self) -> tuple:
        """Get depot coordinates from user with range validation."""
        print("\n" + "=" * 40)
        print("        DEPOT CONFIGURATION")
        print("=" * 40)
        print(f"Current default: ({DEFAULT_CONFIG['depot_lat']}, {DEFAULT_CONFIG['depot_lon']})")
        print("This is the start and end point for all routes.")

        while True:
            try:
                use_default = input("Use default depot? (y/n): ").strip().lower()
                if use_default in ['y', 'yes', '']:
                    return DEFAULT_CONFIG['depot_lat'], DEFAULT_CONFIG['depot_lon']
                elif use_default in ['n', 'no']:
                    print("\nEnter custom depot coordinates:")
                    lat = float(input("Latitude (e.g., 59.911): "))
                    lon = float(input("Longitude (e.g., 10.750): "))
                    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                        print("Coordinates out of range. Latitude must be -90..90 and Longitude -180..180.")
                        continue
                    return lat, lon
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
            except ValueError:
                print("Invalid coordinate. Please enter numeric values.")
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled.")
                sys.exit(0)

    def get_input_file(self) -> str:
        """Prompt for input CSV path with default."""
        print("\n" + "=" * 40)
        print("        INPUT DATA FILE")
        print("=" * 40)
        default = DEFAULT_CONFIG['input_file']
        path = input(f"Input CSV path [{default}]: ").strip()
        path = path or default
        return path

    def run_text_menu(self) -> Dict[str, Any]:
        """Run the interactive text menu interface."""
        self.display_welcome()

        # Get transport mode
        transport_mode = self.display_transport_menu()

        # Get optimization objective
        objective = self.display_objective_menu()

        # Get depot coordinates
        depot_lat, depot_lon = self.get_depot_coordinates()

        # Get input file
        input_file = self.get_input_file()

        # Confirm configuration
        print("\n" + "=" * 40)
        print("        CONFIRM CONFIGURATION")
        print("=" * 40)
        print(f"Transport Mode:    {transport_mode}")
        print(f"Objective:         {objective}")
        print(f"Depot Location:    ({depot_lat}, {depot_lon})")
        print(f"Input File:        {input_file}")
        print("=" * 40)

        while True:
            try:
                confirm = input("\nStart optimization with these settings? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    return {
                        'input_file': input_file,
                        'depot_lat': depot_lat,
                        'depot_lon': depot_lon,
                        'mode': transport_mode,
                        'objective': objective
                    }
                elif confirm in ['n', 'no']:
                    print("Restarting configuration...")
                    return self.run_text_menu()
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled.")
                sys.exit(0)

    def display_summary(self, results: Dict[str, Any], config: Dict[str, Any]):
        """Display optimization results summary."""
        print("\n" + "=" * 60)
        print("           OPTIMIZATION RESULTS SUMMARY")
        print("=" * 60)

        metrics = results.get('final_metrics', {
            'total_distance_km': 0.0,
            'total_time_hours': 0.0,
            'total_cost_nok': 0.0,
            'total_co2_g': 0.0
        })
        tour = results.get('optimized_tour', [])

        transport_mode_name = results.get('transport_mode', config.get('mode', 'Unknown'))
        objective_name = results.get('objective', config.get('objective', 'Unknown'))
        improvement_pct = results.get('improvement_pct', 0.0)

        print(f"Transport Mode:    {transport_mode_name}")
        print(f"Optimization:      {objective_name.upper()}")
        # Exclude depot start/end if present (safe guard)
        deliveries_count = max(0, len(tour) - 2) if len(tour) >= 2 else len(tour)
        print(f"Deliveries:        {deliveries_count}")
        print(f"Improvement:       {improvement_pct:.2f}%")

        print("\n--- PERFORMANCE METRICS ---")
        print(f"Total Distance:    {metrics.get('total_distance_km', 0.0):.2f} km")

        # Format time appropriately
        total_hours = float(metrics.get('total_time_hours', 0.0))
        if total_hours < 1:
            minutes = total_hours * 60
            print(f"Total Time:        {minutes:.1f} minutes")
        else:
            print(f"Total Time:        {total_hours:.2f} hours ({total_hours * 60:.1f} minutes)")

        print(f"Total Cost:        {metrics.get('total_cost_nok', 0.0):.2f} NOK")
        print(f"Total CO₂:         {metrics.get('total_co2_g', 0.0):.1f} g")

        print(f"\n--- ROUTE OVERVIEW ---")
        print(f"Stops: {len(tour)} (including depot start/end)")

        print("\nGenerated files:")
        print("  ✓ route.csv - Detailed route information")
        print("  ✓ rejected.csv - Rejected delivery records (if any)")
        print("  ✓ run.log - Execution log with timing")

        print("\n" + "=" * 60)

    def display_pareto_results(self, pareto_solutions: List[Dict[str, Any]], config: Dict[str, Any]):
        """Display Pareto front results and let user select a solution."""
        print("\n" + "=" * 60)
        print("           PARETO OPTIMAL SOLUTIONS")
        print("=" * 60)
        print(f"Transport Mode:    {config['mode']}")
        print(f"Found {len(pareto_solutions)} non-dominated solutions")
        print("Each solution represents a different trade-off between Time, Cost, and CO₂")
        print("=" * 60)

        # Sort solutions by time for display
        pareto_solutions.sort(key=lambda x: x.get('time_hours', float('inf')))

        print("\nSolution | Time (hours) | Cost (NOK) | CO₂ (g) | Weights (T/C/E)")
        print("-" * 70)

        for i, solution in enumerate(pareto_solutions, 1):
            weights = solution.get('weights', {})
            print(
                f"{i:8} | {solution.get('time_hours', 0.0):12.3f} | "
                f"{solution.get('cost_nok', 0.0):10.2f} | "
                f"{solution.get('co2_g', 0.0):7.1f} | "
                f"{weights.get('time_weight', 0.0):.2f}/"
                f"{weights.get('cost_weight', 0.0):.2f}/"
                f"{weights.get('co2_weight', 0.0):.2f}"
            )

        print("\n" + "=" * 60)
        print("KEY: T=Time Weight, C=Cost Weight, E=Emissions Weight")
        print("Weights show the relative importance of each objective in the solution")
        print("No solution can be improved in one objective without worsening another")
        print("=" * 60)

        # Let user select a solution for detailed output
        if pareto_solutions:
            self._select_pareto_solution(pareto_solutions, config)

    def _select_pareto_solution(self, pareto_solutions: List[Dict], config: Dict):
        """Allow user to select a solution from the Pareto front."""
        print(f"\nSelect a solution to generate detailed route (1-{len(pareto_solutions)}):")
        print("Or press Enter to skip and only generate Pareto front CSV.")

        while True:
            try:
                choice = input(f"Enter solution number (1-{len(pareto_solutions)}) or press Enter to skip: ").strip()

                if choice == '':
                    print("Skipping detailed route generation.")
                    return

                solution_num = int(choice)
                if 1 <= solution_num <= len(pareto_solutions):
                    selected_solution = pareto_solutions[solution_num - 1]
                    self._generate_solution_output(selected_solution, config)
                    break
                else:
                    print(f"Please enter a number between 1 and {len(pareto_solutions)}")

            except ValueError:
                print("Please enter a valid number or press Enter to skip")
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled.")
                return

    def _generate_solution_output(self, solution: Dict, config: Dict):
        """Generate output files for selected Pareto solution."""
        print(f"\nGenerating output for Solution {solution.get('solution_id', 'N/A')}...")

        try:
            # Generate the route CSV for the selected solution
            transport_mode = TransportMode.get_mode(config['mode'])
            OutputGenerator.generate_pareto_solution_route(
                solution.get('tour', []),
                solution,
                transport_mode
            )

            print(f"✓ Generated route_solution_{solution.get('solution_id', 'N_A')}.csv")
            print(f"✓ Solution Details:")
            print(f"  - Time: {solution.get('time_hours', 0.0):.2f} hours ({solution.get('time_hours', 0.0) * 60:.1f} minutes)")
            print(f"  - Cost: {solution.get('cost_nok', 0.0):.2f} NOK")
            print(f"  - CO₂:  {solution.get('co2_g', 0.0):.1f} g")
            w = solution.get('weights', {})
            print(f"  - Weights: Time={w.get('time_weight', 0.0):.2f}, "
                  f"Cost={w.get('cost_weight', 0.0):.2f}, "
                  f"CO₂={w.get('co2_weight', 0.0):.2f}")

        except Exception as e:
            print(f"❌ Failed to generate solution output: {e}")
            logger.error(f"Failed to generate Pareto solution output: {e}")


class OutputGenerator:
    """Handles generation of output files."""

    @staticmethod
    def _safe_name_and_priority(stop: Depot) -> Tuple[str, Union[int, str]]:
        name = getattr(stop, 'customer', None)
        if not name:
            # prefer a recognizable label for depots
            name = getattr(stop, 'name', 'Depot')
        priority = getattr(stop, 'priority', '')
        return str(name), priority

    @staticmethod
    def generate_route_csv(tour: List[Depot], results: Dict[str, Any],
                           transport_mode: TransportMode, objective: str) -> None:
        """
        Generate route.csv with detailed stop information.

        Args:
            tour: Optimized tour sequence
            results: Optimization results dictionary
            transport_mode: Selected transport mode
            objective: Optimization objective
        """
        try:
            with open('route.csv', 'w', newline='', encoding='utf-8') as f:
                # Optional header comment
                f.write(f"# Objective: {objective}\n")
                f.write(f"# Transport Mode: {getattr(transport_mode, 'name', str(transport_mode))}\n")
                # Write header
                f.write("stop_number,customer,latitude,longitude,priority,"
                        "distance_from_previous_km,cumulative_distance_km,"
                        "eta_hours,cumulative_cost_nok,cumulative_co2_g\n")

                cumulative_distance = 0.0
                cumulative_time = 0.0
                cumulative_cost = 0.0
                cumulative_co2 = 0.0

                for i in range(len(tour)):
                    current_stop = tour[i]

                    # Calculate segment metrics (except for first stop)
                    segment_distance = 0.0
                    if i > 0:
                        prev_stop = tour[i - 1]

                        # Calculate distance
                        segment_distance = HaversineCalculator.calculate_delivery_distance(
                            prev_stop, current_stop
                        )

                        segment_time = segment_distance / transport_mode.speed_kph
                        segment_cost = segment_distance * transport_mode.cost_per_km
                        segment_co2 = segment_distance * transport_mode.co2_per_km

                        cumulative_distance += segment_distance
                        cumulative_time += segment_time
                        cumulative_cost += segment_cost
                        cumulative_co2 += segment_co2

                    name, priority = OutputGenerator._safe_name_and_priority(current_stop)

                    # Write row
                    f.write(f"{i + 1},"
                            f"\"{name}\","
                            f"{current_stop.latitude:.6f},"
                            f"{current_stop.longitude:.6f},"
                            f"{priority},"
                            f"{segment_distance:.4f},"
                            f"{cumulative_distance:.4f},"
                            f"{cumulative_time:.4f},"
                            f"{cumulative_cost:.2f},"
                            f"{cumulative_co2:.1f}\n")

            logger.info(f"Generated route.csv with {len(tour)} stops")

        except Exception as e:
            logger.error(f"Failed to generate route.csv: {str(e)}")
            raise

    @staticmethod
    def generate_pareto_csv(pareto_solutions: List[Dict[str, Any]], filename: str = "pareto_front.csv") -> None:
        """
        Generate CSV file with Pareto front solutions.

        Args:
            pareto_solutions: List of Pareto-optimal solutions
            filename: Output filename
        """
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                f.write("solution_id,time_weight,cost_weight,co2_weight,"
                        "time_hours,cost_nok,co2_g,distance_km,improvement_pct\n")

                for solution in pareto_solutions:
                    weights = solution.get('weights', {})
                    f.write(f"{solution.get('solution_id', '')},"
                            f"{weights.get('time_weight', 0.0):.3f},"
                            f"{weights.get('cost_weight', 0.0):.3f},"
                            f"{weights.get('co2_weight', 0.0):.3f},"
                            f"{solution.get('time_hours', 0.0):.4f},"
                            f"{solution.get('cost_nok', 0.0):.2f},"
                            f"{solution.get('co2_g', 0.0):.1f},"
                            f"{solution.get('distance_km', 0.0):.2f},"
                            f"{solution.get('improvement_pct', 0.0):.2f}\n")

            logger.info(f"Generated Pareto front CSV: {filename}")

        except Exception as e:
            logger.error(f"Failed to generate Pareto CSV: {str(e)}")
            raise

    @staticmethod
    def generate_pareto_solution_route(tour: List[Depot], solution: Dict[str, Any],
                                       transport_mode: TransportMode) -> None:
        """
        Generate route CSV for a specific Pareto solution.

        Args:
            tour: Optimized tour sequence
            solution: Pareto solution dictionary
            transport_mode: Selected transport mode
        """
        solution_id = solution.get('solution_id', 'N_A')
        filename = f"route_solution_{solution_id}.csv"

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                # Enhanced header with solution info
                weights = solution.get('weights', {})
                f.write("# Pareto Solution Details:\n")
                f.write(f"# Solution ID: {solution_id}\n")
                f.write(f"# Weights - Time: {weights.get('time_weight', 0.0):.2f}, "
                        f"Cost: {weights.get('cost_weight', 0.0):.2f}, "
                        f"CO₂: {weights.get('co2_weight', 0.0):.2f}\n")
                f.write(f"# Total Time: {solution.get('time_hours', 0.0):.2f} hours\n")
                f.write(f"# Total Cost: {solution.get('cost_nok', 0.0):.2f} NOK\n")
                f.write(f"# Total CO₂: {solution.get('co2_g', 0.0):.1f} g\n")
                f.write(f"# Total Distance: {solution.get('distance_km', 0.0):.2f} km\n")
                f.write("#\n")

                # Route data header
                f.write("stop_number,customer,latitude,longitude,priority,"
                        "distance_from_previous_km,cumulative_distance_km,"
                        "eta_hours,cumulative_cost_nok,cumulative_co2_g\n")

                cumulative_distance = 0.0
                cumulative_time = 0.0
                cumulative_cost = 0.0
                cumulative_co2 = 0.0

                for i in range(len(tour)):
                    current_stop = tour[i]

                    # Calculate segment metrics (except for first stop)
                    segment_distance = 0.0
                    if i > 0:
                        prev_stop = tour[i - 1]
                        segment_distance = HaversineCalculator.calculate_delivery_distance(
                            prev_stop, current_stop
                        )

                        segment_time = segment_distance / transport_mode.speed_kph
                        segment_cost = segment_distance * transport_mode.cost_per_km
                        segment_co2 = segment_distance * transport_mode.co2_per_km

                        cumulative_distance += segment_distance
                        cumulative_time += segment_time
                        cumulative_cost += segment_cost
                        cumulative_co2 += segment_co2

                    name, priority = OutputGenerator._safe_name_and_priority(current_stop)

                    # Write row
                    f.write(f"{i + 1},"
                            f"\"{name}\","
                            f"{current_stop.latitude:.6f},"
                            f"{current_stop.longitude:.6f},"
                            f"{priority},"
                            f"{segment_distance:.4f},"
                            f"{cumulative_distance:.4f},"
                            f"{cumulative_time:.4f},"
                            f"{cumulative_cost:.2f},"
                            f"{cumulative_co2:.1f}\n")

            logger.info(f"Generated Pareto solution route: {filename}")

        except Exception as e:
            logger.error(f"Failed to generate Pareto solution route: {str(e)}")
            raise


@timing_decorator()
def run_optimization(input_file: str, depot_lat: float, depot_lon: float,
                     mode: str, objective: str) -> Dict[str, Any]:
    """
    Main optimization function with timing decorator.

    Args:
        input_file: Path to input CSV file
        depot_lat: Depot latitude
        depot_lon: Depot longitude
        mode: Transport mode name
        objective: Optimization objective

    Returns:
        Dictionary with optimization results
    """
    logger.info(
        f"Starting optimization: file={input_file}, depot=({depot_lat}, {depot_lon}), "
        f"mode={mode}, objective={objective}"
    )

    # Step 1: Data Validation
    logger.info("Step 1: Validating input data...")
    try:
        valid_deliveries, rejected_rows = DataValidator.validate_input(input_file)
        logger.info(f"Validation complete: {len(valid_deliveries)} valid, "
                    f"{len(rejected_rows)} rejected deliveries")

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        print(f"\n❌ ERROR: Input file '{input_file}' not found.")
        print("Please check the file path and try again.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        print(f"\n❌ ERROR: Data validation failed: {e}")
        sys.exit(1)

    # Check if we have any valid deliveries
    if not valid_deliveries:
        logger.error("No valid deliveries found after validation")
        print("\n❌ ERROR: No valid deliveries found in the input file.")
        print("Check rejected.csv for details on why deliveries were rejected.")
        sys.exit(1)

    # Step 2: Setup Depot and Transport Mode
    logger.info("Step 2: Setting up depot and transport mode...")
    try:
        depot = Depot(depot_lat, depot_lon)
        transport_mode = TransportMode.get_mode(mode)

        logger.info(f"Depot: {depot}")
        logger.info(f"Transport: {transport_mode}")

    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        print(f"\n❌ ERROR: Setup failed: {e}")
        sys.exit(1)

    # Step 3: Run Optimization
    logger.info("Step 3: Running route optimization...")

    objective_norm = (objective or '').strip().lower()

    if objective_norm == 'pareto':
        # Run Pareto multi-objective optimization
        try:
            pareto_optimizer = ParetoOptimizer(depot, valid_deliveries, transport_mode)
            pareto_solutions = pareto_optimizer.generate_pareto_front(num_solutions=8)

            # Generate Pareto front CSV
            OutputGenerator.generate_pareto_csv(pareto_solutions)
            logger.info("Pareto optimization complete")

            return {
                'pareto_solutions': pareto_solutions,
                'transport_mode': transport_mode.name,
                'objective': 'Pareto'
            }

        except Exception as e:
            logger.error(f"Pareto optimization failed: {str(e)}")
            print(f"\n❌ ERROR: Pareto optimization failed: {e}")
            sys.exit(1)
    else:
        # Run single-objective optimization
        try:
            # Capitalize objective to align with menus ('Time','Cost','CO2')
            normalized_objective = objective_norm.upper()
            if normalized_objective == 'CO2':
                normalized_objective = 'CO2'
            elif normalized_objective == 'TIME':
                normalized_objective = 'Time'
            elif normalized_objective == 'COST':
                normalized_objective = 'Cost'

            optimizer = RouteOptimizer(depot, valid_deliveries, transport_mode, normalized_objective)
            optimized_tour, results = optimizer.optimize_route()

            # Ensure UI-required fields exist
            results = results or {}
            results.setdefault('optimized_tour', optimized_tour)
            results.setdefault('transport_mode', transport_mode.name)
            results.setdefault('objective', normalized_objective)
            results.setdefault('final_metrics', results.get('final_metrics', {
                'total_distance_km': 0.0,
                'total_time_hours': 0.0,
                'total_cost_nok': 0.0,
                'total_co2_g': 0.0,
            }))
            results.setdefault('improvement_pct', results.get('improvement_pct', 0.0))

            logger.info(f"Optimization successful: {len(optimized_tour)} stops in final tour")

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            print(f"\n❌ ERROR: Route optimization failed: {e}")
            sys.exit(1)

    # Step 4: Generate Output
    logger.info("Step 4: Generating output files...")
    try:
        if objective_norm != 'pareto':
            OutputGenerator.generate_route_csv(results['optimized_tour'], results, transport_mode, results.get('objective', ''))
        logger.info("Output generation complete")

    except Exception as e:
        logger.error(f"Output generation failed: {str(e)}")
        print(f"\n❌ ERROR: Failed to generate output files: {e}")
        sys.exit(1)

    return results if objective_norm != 'pareto' else {
        'pareto_solutions': [],  # Safety fallback
        'transport_mode': transport_mode.name,
        'objective': 'Pareto'
    }


def main():
    """Main entry point of the application."""
    interface = CourierOptimizerInterface()

    # Parse command line arguments
    args = interface.parse_arguments()

    if args.version:
        print("CourierOptimizer 1.0.0")
        return

    # Handle list-modes option
    if args.list_modes:
        print("\nAvailable Transport Modes:")
        for mode_name, mode in TransportMode.get_modes_config().items():
            # tolerate dict or object
            speed = getattr(mode, 'speed_kph', None)
            cost = getattr(mode, 'cost_per_km', None)
            co2 = getattr(mode, 'co2_per_km', None)
            if speed is None and isinstance(mode, dict):
                speed = mode.get('speed_kph')
                cost = mode.get('cost_per_km')
                co2 = mode.get('co2_per_km')
            print(f"  {mode_name}:")
            print(f"    Speed:    {speed} km/h")
            print(f"    Cost:     {cost} NOK/km")
            print(f"    CO₂:      {co2} g/km")
        return

    # Determine interface mode
    use_text_menu = args.text_menu or (args.mode is None or args.objective is None)

    if use_text_menu:
        # Use text menu interface
        config = interface.run_text_menu()
    else:
        # Use CLI interface
        config = {
            'input_file': args.input_file,
            'depot_lat': args.depot[0],
            'depot_lon': args.depot[1],
            'mode': args.mode,
            'objective': args.objective
        }

        # Display configuration
        interface.display_welcome()
        print(f"\nConfiguration:")
        print(f"  Input file:    {config['input_file']}")
        print(f"  Depot:         ({config['depot_lat']}, {config['depot_lon']})")
        print(f"  Transport:     {config['mode']}")
        print(f"  Objective:     {config['objective']}")
        print(f"\nStarting optimization...")

    # Check if input file exists (user feedback early)
    if not os.path.exists(config['input_file']):
        print(f"\n❌ ERROR: Input file '{config['input_file']}' not found.")
        print("Please check the file path or use the -i option to specify a different file.")
        sys.exit(1)

    try:
        # Run the main optimization process
        _result = run_optimization(**config)
        # Be tolerant of timing_decorator returning (result, elapsed)
        results = _result[0] if isinstance(_result, tuple) else _result

        # Display appropriate results based on optimization type
        if (config['objective'] or '').strip().lower() == 'pareto':
            interface.display_pareto_results(results.get('pareto_solutions', []), config)
        else:
            interface.display_summary(results, config)

        # Success exit
        sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        print("\n\n⚠️  Optimization interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\n\n❌ UNEXPECTED ERROR: {e}")
        print("Check run.log for detailed error information.")
        sys.exit(1)


if __name__ == '__main__':
    main()