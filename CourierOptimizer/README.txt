Part1: Smart courier routing

CourierOptimizer
----------------

CourierOptimizer is a small command-line tool for optimizing last-mile delivery routes around Oslo. 
It reads delivery jobs from a CSV file, validates the data, and thencomputes efficient routes for 
different transport modes (Car, Bicycle, Walking), with support for both single-objective optimization 
(Time, Cost, CO₂) and multi-objective Pareto optimization. The tool also produces CSV reports for 
the optimized route, rejected input rows, and (optionally) a Pareto front of trade-off solutions.


1. Features:

- CSV input validation with clear error messages and a rejected-rows report.
- Strong data checks:
  - Required columns and non-empty values.
  - Numeric parsing (including scientific notation and decimal commas).
  - Geographic bounds restricted to the Oslo area.
  - Weight limits and allowed priority levels.
- Core data models:
  - Deliveries with priority and weight.
  - Depot as start/end point.
  - Transport modes with speed, cost, and CO₂ per km.
- Route optimization:
  - Haversine distance calculations.
  - Initial tour via Nearest Neighbor.
  - Local search improvement via 2-opt.
  - Objective-aware weighted costs (Time, Cost, CO₂), including priority multipliers.
- Multi-objective optimization:
  - Weighted combinations of Time/Cost/CO₂.
  - Automatic generation of Pareto-optimal (non-dominated) solutions.
- Output reports:
  - Detailed route CSV with distances, ETA, cumulative cost, and CO₂.
  - Pareto front CSV with weight settings and resulting metrics.
  - Rejected input rows with row number and rejection reason.
  - Execution log with timing information.
- Extensibility:
  - Dynamically add new transport modes at runtime.
  - Dynamically add new priority levels with custom multipliers.
- Two interfaces:
  - Interactive text menu.
  - Direct CLI mode with arguments (suitable for scripts/automation).


2. Project Structure:

- `main.py`
  - Entry point of the application.
  - Implements `CourierOptimizerInterface`:
    - Argument parsing (`argparse`).
    - Interactive text-menu interface.
    - Result summaries and Pareto solution selection.
  - Implements `OutputGenerator`:
    - `generate_route_csv(...)`
    - `generate_pareto_csv(...)`
    - `generate_pareto_solution_route(...)`
  - Defines `run_optimization(...)` (decorated with a timing logger).
  - Defines `main()` which wires everything together for CLI/text-menu use.

- `data_validator.py`
  - Defines `DataValidator`, responsible for validating CSV input data.
  - Separates valid deliveries from rejected rows and writes `rejected.csv`.

- `models.py`
  - Core data models:
    - `InvalidDataError` (custom exception for validation issues).
    - `Priority` enum (`HIGH`, `MEDIUM`, `LOW`) with multipliers.
    - `TransportMode` (Car, Bicycle, Walking by default).
    - `Delivery` (customer stop).
    - `Depot` (inherits from `Delivery` and marks `is_depot = True`).

- `optimization.py`
  - `RouteOptimizer`:
    - Builds a distance matrix using Haversine.
    - Generates an initial tour via Nearest Neighbor.
    - Improves the route with 2-opt local search.
    - Computes metrics for distance, time, cost, and CO₂.
  - `WeightedRouteOptimizer`:
    - Extends `RouteOptimizer` with configurable weights for Time/Cost/CO₂.
  - `ParetoOptimizer`:
    - Generates multiple weight combinations.
    - Runs optimizations to gather candidate solutions.
    - Filters the Pareto front (non-dominated solutions).

- `utilities.py`
  - `HaversineCalculator`:
    - Computes great-circle distances in kilometers.
    - Also provides a helper for distance between `Delivery` objects.
  - `timing_decorator(...)`:
    - Decorator to log execution time and parameters.
    - Writes summary lines to a log file (by default `run.log`).
  - `DynamicConfigLoader`:
    - `add_priority_level(...)` to inject new priority multipliers into a class.
    - `add_transport_mode(...)` to extend `TransportMode.get_modes_config()` at runtime.


3. Requirements & Installation:

- Python 3.10+ (recommended; project uses type hints and `dataclasses`).
- Standard library only:
  - `argparse`, `csv`, `logging`, `math`, `time`, `functools`, `inspect`, `re`, etc.
- No external dependencies are required.

To use the tool, simply place all `.py` files in the same folder and run:

    python main.py

Optionally, set up a virtual environment if you want to keep things isolated,
though it is not strictly necessary since there are no third-party packages.


4. Input Data Format:

The optimizer expects a CSV file with the following header columns (case-insensitive
but must be present):

    customer, latitude, longitude, priority, weight_kg

Each row represents a single delivery stop:

- `customer`:
  - Required, non-empty.
  - Must match a simple pattern: alphanumeric characters, spaces, hyphens,
    and dots are allowed.
- `latitude`, `longitude`:
  - Required, numeric (supports scientific notation and decimal commas).
  - Global bounds:
    - Latitude:  -90 .. 90
    - Longitude: -180 .. 180
  - Oslo-area bounds (stricter):
    - Latitude:  59.0  .. 60.5
    - Longitude: 10.0  .. 11.5
- `priority`:
  - Required.
  - Must be one of: `High`, `Medium`, `Low` (case-insensitive in the CSV).
  - Internally mapped to the `Priority` enum.
- `weight_kg`:
  - Required, numeric.
  - Must be ≥ 0.
  - Must not exceed 50 kg (configurable constant in `DataValidator`).

Additional behavior:

- Fully empty lines are skipped.
- Inline comments in numeric fields are supported:
  - Example: `5.0  # Invalid latitude` – the ` # ...` part is stripped before parsing.
- Decimal commas are normalized:
  - Example: `"2,5"` is treated as `2.5`.

Invalid rows are not fatal for the whole run. Instead, they are:

- Logged with a reason.
- Collected and written to `rejected.csv` together with:
  - Original columns.
  - Row number.
  - Human-readable `rejection_reason`.


5. Running the Optimizer:

You can run CourierOptimizer in two main ways.

A) Interactive text-menu interface

Run:

    python main.py

If you do not pass `--mode` and `--objective`, the program shows a text menu:

1. Select transport mode:
   - Car
   - Bicycle
   - Walking
   (also any dynamically added modes such as Scooter, if configured at runtime)

2. Select optimization objective:
   - Time (fastest route)
   - Cost (lowest cost)
   - CO₂ (lowest emissions)
   - Pareto (multi-objective trade-off exploration)

3. Configure depot coordinates:
   - Use default depot (e.g., Oslo center), or
   - Enter custom latitude and longitude (with validation).

4. Choose input CSV file path:
   - Defaults to `input_data.csv`, or you can enter a custom path.

The configuration is then confirmed on screen, and you can start the optimization.

B) CLI usage with arguments:

You can also run everything non-interactively, suitable for scripts or batch jobs:

    python main.py -i deliveries.csv -m Car -o Time
    python main.py -i deliveries.csv -m Bicycle -o Cost
    python main.py -i deliveries.csv -m Car -o Pareto --depot 59.912 10.749

Supported options:

- `-i`, `--input`:
  - Path to the CSV file (default: `input_data.csv`).
- `-m`, `--mode`:
  - Transport mode; must be one of the available modes.
- `-o`, `--objective`:
  - `Time`, `Cost`, `CO2`, or `Pareto`.
- `--depot LAT LON`:
  - Custom depot coordinates.
- `--list-modes`:
  - Print available transport modes and their speed/cost/CO₂, then exit.
- `--text-menu`:
  - Force use of the interactive text menu.
- `--version`:
  - Print the program version and exit.

Before running the optimization, the program checks that the input file exists and
prints a user-friendly error message if not.


6. Optimization Logic:

1) Data validation
   - `DataValidator.validate_input(path)`:
     - Reads the CSV.
     - Validates headers and each row.
     - Returns:
       - `valid_deliveries`: list of `Delivery` objects (using the `Priority` enum).
       - `rejected_rows`: list of dicts (also written to `rejected.csv`).

2) Model setup
   - A `Depot` object is created from the depot coordinates.
   - A `TransportMode` is selected based on the chosen mode name.

3) Route optimization
   - `RouteOptimizer` is initialized with:
     - Depot, list of deliveries, transport mode, and objective.
   - Distance matrix:
     - Pre-computed pairwise distances between all stops using Haversine.
   - Initial tour:
     - Computed via Nearest Neighbor (starting from the depot).
   - Local search:
     - 2-opt is applied to the index-based tour representation.
     - The algorithm iteratively performs 2-opt swaps while an improvement in
       “weighted cost” is found.
   - Cost and metrics:
     - For each leg, the following is accumulated:
       - Distance (km).
       - Time (hours) = distance / speed_kph.
       - Cost (NOK) = distance * cost_per_km.
       - CO₂ (g)   = distance * co2_per_km.
     - The “weighted cost” used for optimization depends on the chosen objective:
       - Time:   distance / speed (hours).
       - Cost:   distance * cost_per_km.
       - CO₂:    distance * co2_per_km.
     - A priority multiplier is applied at the destination stop:
       - High priority → smaller multiplier (e.g., 0.6).
       - Medium priority → 1.0.
       - Low priority → 1.2.
   - Results include:
     - Initial and final tour.
     - Initial and final costs.
     - Percentage improvement.
     - Final metrics for distance, time, cost, and CO₂.

4) Pareto optimization (multi-objective)
   - When objective is `Pareto`:
     - `ParetoOptimizer` generates several weight combinations for Time, Cost, and CO₂.
     - For each combination:
       - `WeightedRouteOptimizer` is used to run a full optimization.
       - The resulting solution stores:
         - Weights.
         - Final time, cost, CO₂, distance.
         - Improvement percentage.
         - Tour.
     - A Pareto filter keeps only non-dominated solutions:
       - No other solution is strictly better in all objectives.
     - These Pareto solutions are:
       - Written to `pareto_front.csv`.
       - Displayed in a table.
       - Optionally used to generate a dedicated route CSV for a user-selected solution.


7. Output Files

Depending on the chosen objective and the outcome, the following files may be created:

1) `route.csv` (single-objective runs)
   - Contains detailed route information for the optimized tour:
   - Columns (in order):
     - `stop_number`
     - `customer`
     - `latitude`
     - `longitude`
     - `priority`
     - `distance_from_previous_km`
     - `cumulative_distance_km`
     - `eta_hours`
     - `cumulative_cost_nok`
     - `cumulative_co2_g`
   - Includes both depot and delivery stops.

2) `rejected.csv`
   - Contains rejected input rows and reasons:
     - `customer, latitude, longitude, priority, weight_kg, row_number, rejection_reason`

3) `pareto_front.csv` (Pareto runs)
   - Contains one row per Pareto-optimal solution:
     - `solution_id`
     - `time_weight`
     - `cost_weight`
     - `co2_weight`
     - `time_hours`
     - `cost_nok`
     - `co2_g`
     - `distance_km`
     - `improvement_pct`

4) `route_solution_<ID>.csv` (optional, Pareto runs)
   - Generated when the user selects a specific Pareto solution in the menu.
   - Same structure as `route.csv`, but annotated with solution metadata in comments.

5) `run.log`
   - Plain-text log capturing:
     - Start/stop timestamps.
     - Execution time of `run_optimization`.
     - Errors, if any.
   - Also used by `timing_decorator` to append SUCCESS/ERROR lines.


8. Logging & Error Handling:

- Logging:
  - Uses Python’s `logging` module.
  - Main logger writes to:
    - Standard output (console).
    - `run.log` file (overwritten each run in `main.py`).
- Common error conditions:
  - Missing input file → user-friendly message and exit code 1.
  - No valid deliveries after validation → message directing user to `rejected.csv`.
  - Unknown transport mode name → `ValueError` with a clear list of valid modes.
  - Unexpected internal errors:
    - Logged with full details.
    - Printed as a summary with advice to inspect `run.log`.


9. Extensibility:


Dynamic priority levels:


- You can dynamically add/update priority categories at runtime using:

    DynamicConfigLoader.add_priority_level(Delivery, "Urgent", 0.3)

- This:
  - Ensures a `_priority_multipliers` dict exists on the class.
  - Sets/injects a `get_priority_multiplier` instance method that looks up
    the multiplier based on the `priority` attribute.

Dynamic transport modes:


- You can dynamically add a new transport mode:

    DynamicConfigLoader.add_transport_mode("Scooter", 25.0, 1.5, 15.0)

- Internally:
  - The existing `TransportMode.get_modes_config()` is wrapped so that it
    returns a fresh dict including the new mode.
  - This new mode will be visible in `--list-modes` and selectable by name.


10. Testing & Examples:


Each module includes a small `if __name__ == "__main__":` block that can be used
for manual testing:

- `models.py`:
  - Demonstrates:
    - Raising and printing `InvalidDataError`.
    - Listing available transport modes.
    - Creating `Delivery` and `Depot` instances.

- `utilities.py`:
  - Demonstrates:
    - Haversine distance between Oslo and Bergen.
    - `timing_decorator` on a simple sleep function.
    - Error logging via `timing_decorator`.
    - Dynamic addition of a new priority level and transport mode.

- `optimization.py`:
  - Demonstrates:
    - Building a simple test scenario with a handful of deliveries.
    - Running `RouteOptimizer` for objectives `Time`, `Cost`, `CO2`.
    - Printing optimized tours and metrics.
    - Running multi-objective Pareto optimization and printing solutions.


11. License & Author


- Author: <Syed Fawad Ali Shah>
- Email:  <sysha6706@oslomet.no>
- License: OsloMet