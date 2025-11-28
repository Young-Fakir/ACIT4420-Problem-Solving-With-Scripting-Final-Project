"""
main.py - Conway's Game of Life Simulator

Main orchestrator module that integrates all components:
- Board management
- Rule evolution
- Pattern parsing
- File I/O operations
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from board import Board
from rules import RuleEngine, list_available_rules
from pattern_parser import PatternParser, PatternParseError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("game_of_life.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
_logger = logging.getLogger(__name__)


class GameOfLifeSimulator:
    """
    Main orchestrator class for Conway's Game of Life simulation.
    Holds history/backtracking locally and evolves via RuleEngine.
    """

    def __init__(self, width: int = 50, height: int = 50):
        """
        Initialize the simulator with specified grid dimensions.

        Args:
            width: Grid width
            height: Grid height
        """
        self.board = Board(width, height)
        self.rule_engine = RuleEngine()
        self.pattern_parser = PatternParser()
        self.is_running = False

        # History of previous grids (np.ndarray[bool]) for backtracking/stability
        self._history: List[np.ndarray] = []

        self.simulation_stats = {
            "total_generations": 0,
            "patterns_loaded": 0,
            "snapshots_saved": 0,
        }

        _logger.info("Initialized Game of Life simulator (%dx%d)", width, height)

    def load_pattern(self, pattern_file: str) -> bool:
        """
        Load a pattern from file onto the board.

        Args:
            pattern_file: Path to pattern file

        Returns:
            True if successful, False otherwise
        """
        try:
            live_cells = self.pattern_parser.load_pattern(pattern_file)

            # Clear board and apply pattern
            self.board.clear()
            # overlay application
            for x, y in live_cells:
                try:
                    self.board.set_cell(x, y, 1)
                except Exception as e:
                    _logger.warning("Skipping invalid cell (%d, %d): %s", x, y, e)

            # Reset generation and history after a fresh load
            self.board.generation = 0
            self._history.clear()

            self.simulation_stats["patterns_loaded"] += 1
            _logger.info("Successfully loaded pattern from %s", pattern_file)
            return True

        except (PatternParseError, FileNotFoundError) as e:
            _logger.error("Failed to load pattern: %s", e)
            return False

    def evolve_generation(self) -> None:
        """
        Evolve the board by one generation using the RuleEngine.
        Keeps a copy in history for backtracking / stability checks.
        """
        try:
            # Save current state for backtracking
            self._history.append(self.board.grid.copy())

            # Evolve via rule engine; convert back to bool for the Board
            next_grid_int = self.rule_engine.evolve_grid(self.board.grid)
            self.board.grid = next_grid_int.astype(bool, copy=False)

            # Increment counters
            self.board.generation += 1
            self.simulation_stats["total_generations"] += 1

        except Exception as e:
            _logger.error("Evolution failed: %s", e)
            # If evolution failed, drop the pushed history to stay consistent
            if self._history:
                self._history.pop()
            raise

    def run_simulation(self, generations: int, auto_display: bool = True) -> None:
        """
        Run the simulation for a specified number of generations.

        Args:
            generations: Number of generations to simulate
            auto_display: Whether to display each generation
        """
        _logger.info("Starting simulation for %d generations", generations)
        self.is_running = True

        try:
            for _ in range(generations):
                if not self.is_running:
                    break

                # Stop if stable before evolving further (optional choice)
                if self._is_stable_state():
                    _logger.info("Stable state reached, stopping simulation")
                    break

                self.evolve_generation()

                if auto_display:
                    print(f"Generation {self.board.generation}:")
                    self.board.display()
                    print()

                # Or detect stability after evolving (alternate policy)
                if self._is_stable_state():
                    _logger.info("Stable state reached, stopping simulation")
                    break

        except KeyboardInterrupt:
            _logger.info("Simulation interrupted by user")
        except Exception as e:
            _logger.error("Simulation error: %s", e)
        finally:
            self.is_running = False
            _logger.info("Simulation completed")

    def _is_stable_state(self) -> bool:
        """
        Check if the simulation has reached a stable state.
        True if the current grid equals the last historical grid.
        """
        if not self._history:
            return False
        # Compare current to previous generation
        prev = self._history[-1]
        # Shapes identical by construction; compare bytes for speed
        return self.board.grid.shape == prev.shape and (
            self.board.grid.dtype == prev.dtype
        ) and (self.board.grid.tobytes() == prev.tobytes())

    def set_rule(self, rule_name: str) -> bool:
        """
        Change the rule set for evolution.

        Args:
            rule_name: Name of rule set to use

        Returns:
            True if successful, False otherwise
        """
        try:
            self.rule_engine.set_rule(rule_name)
            _logger.info("Rule set changed to: %s", rule_name)
            return True
        except Exception as e:
            _logger.error("Failed to set rule '%s': %s", rule_name, e)
            return False

    def set_boundary(self, boundary: str) -> bool:
        """
        Change boundary conditions.

        Args:
            boundary: 'fixed' or 'toroidal'

        Returns:
            True if successful, False otherwise
        """
        try:
            self.rule_engine.set_boundary(boundary)
            _logger.info("Boundary condition set to: %s", boundary)
            return True
        except Exception as e:
            _logger.error("Failed to set boundary '%s': %s", boundary, e)
            return False

    def save_snapshot(self, filename: str, format: str = "text") -> bool:
        """
        Save current board state to file.

        Args:
            filename: Output file path
            format: Output format ('text' or 'binary')

        Returns:
            True if successful, False otherwise
        """
        try:
            self.board.save_snapshot(filename, format)
            self.simulation_stats["snapshots_saved"] += 1
            _logger.info("Snapshot saved to %s", filename)
            return True
        except Exception as e:
            _logger.error("Failed to save snapshot: %s", e)
            return False

    def backtrack(self) -> bool:
        """
        Revert to previous generation, if available.
        """
        if not self._history:
            _logger.warning("No history available for backtracking")
            return False

        last = self._history.pop()
        self.board.grid = last
        if self.board.generation > 0:
            self.board.generation -= 1
        _logger.info("Backtracked to previous generation")
        return True

    def get_simulation_info(self) -> Dict[str, Any]:
        """
        Get comprehensive simulation information.
        """
        board_stats = self.board.get_statistics()
        rule_info = self.rule_engine.get_rule_info()

        return {
            "board_dimensions": f"{self.board.width}x{self.board.height}",
            "current_generation": self.board.generation,
            "rule_set": rule_info["rule_name"],
            "boundary_condition": rule_info["boundary"],
            "board_statistics": board_stats,
            "simulation_statistics": self.simulation_stats,
            "available_rules": rule_info["available_rules"],
        }


def create_sample_patterns() -> None:
    """Create sample pattern files for demonstration."""
    samples = {
        "glider.rle": "#N Glider\nx = 3, y = 3\nbo$2bo$3o!\n",
        "blinker.txt": "OOO\n",
        "glider_plain.txt": "# Glider pattern\n.O.\n..O\nOOO\n",
        "block.txt": "OO\nOO\n",
    }

    for filename, content in samples.items():
        if not Path(filename).exists():
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            _logger.info("Created sample pattern: %s", filename)


def print_banner() -> None:
    """Print welcome banner."""
    print("\n" + "=" * 60)
    print("        CONWAY'S GAME OF LIFE SIMULATOR")
    print("=" * 60)
    print("Available commands:")
    print("  load [file]     - Load pattern from file or choose from a list")
    print("  rule <name>     - Change rule set")
    print("  boundary <type> - Set boundary (fixed/toroidal)")
    print("  run <gens>      - Run simulation for N generations")
    print("  step            - Evolve one generation")
    print("  display         - Show current board")
    print("  save <file>     - Save current state")
    print("  backtrack       - Go back to previous generation")
    print("  info            - Show simulation information")
    print("  quit            - Exit simulator")
    print("=" * 60)


def find_pattern_files() -> List[str]:
    """
    Find pattern files in the current directory.

    Returns:
        Sorted list of filenames with known pattern extensions.
    """
    exts = {".rle", ".txt", ".lif", ".life"}
    files = []
    for p in Path(".").iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p.name)
    return sorted(files)


def choose_pattern_interactively() -> str:
    """
    Show a numbered list of pattern files and ask the user to choose one.

    Returns:
        Selected filename, or empty string if cancelled.
    """
    files = find_pattern_files()
    if not files:
        print("No pattern files (.rle, .txt, .lif, .life) found in this folder.")
        return ""

    print("\nAvailable pattern files:")
    for idx, name in enumerate(files, start=1):
        print(f"  {idx}. {name}")
    print("  c. cancel")

    while True:
        choice = input("Choose a pattern number (or 'c' to cancel): ").strip().lower()
        if choice in ("c", "q", "quit"):
            return ""
        if choice.isdigit():
            n = int(choice)
            if 1 <= n <= len(files):
                return files[n - 1]
        print("Invalid choice. Please enter a valid number or 'c' to cancel.")


def interactive_mode() -> None:
    """Run the simulator in interactive mode."""
    simulator = GameOfLifeSimulator(20, 20)

    print_banner()
    create_sample_patterns()

    while True:
        try:
            command = input("\n>>> ").strip().split()
            if not command:
                continue

            cmd = command[0].lower()

            if cmd in ("quit", "exit"):
                print("Thanks for playing!")
                break

            elif cmd == "load":
                if len(command) > 1:
                    # User specified a filename directly
                    filename = command[1]
                else:
                    # Show menu of available pattern files
                    filename = choose_pattern_interactively()
                    if not filename:
                        # cancelled
                        continue

                if simulator.load_pattern(filename):
                    print(f"Pattern loaded from {filename}")
                    simulator.board.display()
                else:
                    print(f"Failed to load pattern from {filename}")

            elif cmd == "rule" and len(command) > 1:
                rule_name = command[1]
                available_rules = list_available_rules()
                if rule_name in available_rules:
                    simulator.set_rule(rule_name)
                    print(f"Rule set to: {rule_name}")
                else:
                    print(f"Unknown rule: {rule_name}")
                    print(f"Available rules: {', '.join(available_rules)}")

            elif cmd == "boundary" and len(command) > 1:
                boundary = command[1].lower()
                if boundary in ["fixed", "toroidal"]:
                    simulator.set_boundary(boundary)
                    print(f"Boundary set to: {boundary}")
                else:
                    print("Boundary must be 'fixed' or 'toroidal'")

            elif cmd == "run" and len(command) > 1:
                try:
                    generations = int(command[1])
                    simulator.run_simulation(generations)
                except ValueError:
                    print("Please specify a valid number of generations")

            elif cmd == "step":
                simulator.evolve_generation()
                print(f"Generation {simulator.board.generation}:")
                simulator.board.display()

            elif cmd == "display":
                simulator.board.display()

            elif cmd == "save" and len(command) > 1:
                filename = command[1]
                if simulator.save_snapshot(filename):
                    print(f"State saved to {filename}")
                else:
                    print(f"Failed to save state to {filename}")

            elif cmd == "backtrack":
                if simulator.backtrack():
                    print("Backtracked to previous generation")
                    simulator.board.display()
                else:
                    print("No history available for backtracking")

            elif cmd == "info":
                info = simulator.get_simulation_info()
                print("\nSimulation Information:")
                print(f"  Board: {info['board_dimensions']}")
                print(f"  Generation: {info['current_generation']}")
                print(f"  Rule: {info['rule_set']}")
                print(f"  Boundary: {info['boundary_condition']}")
                print(f"  Live cells: {info['board_statistics']['live_cells']}")
                print(
                    f"  Patterns loaded: {info['simulation_statistics']['patterns_loaded']}"
                )

            elif cmd == "help":
                print_banner()

            else:
                print("Unknown command. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\nUse 'quit' to exit or 'help' for commands.")
        except Exception as e:
            print(f"Error: {e}")


def batch_mode(
    pattern_file: str,
    generations: int,
    rule: str = "conway",
    width: int = 50,
    height: int = 50,
    boundary: str = "fixed",
) -> None:
    """
    Run simulation in batch mode with specified parameters.
    """
    simulator = GameOfLifeSimulator(width, height)

    # Load pattern
    if not simulator.load_pattern(pattern_file):
        print(f"Failed to load pattern from {pattern_file}")
        return

    # Set rule and boundary if specified
    if rule:
        simulator.set_rule(rule)
    if boundary:
        simulator.set_boundary(boundary)

    print(f"Running simulation for {generations} generations...")
    print("Initial state:")
    simulator.board.display()

    # Run simulation
    simulator.run_simulation(generations, auto_display=True)

    # Save final state
    simulator.save_snapshot("final_state.txt")
    print(f"\nFinal state saved to 'final_state.txt'")

    # Print summary
    info = simulator.get_simulation_info()
    print(f"\nSimulation completed:")
    print(f"  Final generation: {info['current_generation']}")
    print(f"  Live cells: {info['board_statistics']['live_cells']}")


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Conway's Game of Life Simulator")
    parser.add_argument("--pattern", "-p", help="Pattern file to load")
    parser.add_argument(
        "--generations",
        "-g",
        type=int,
        default=10,
        help="Number of generations to simulate",
    )
    parser.add_argument(
        "--rule",
        "-r",
        default="conway",
        help="Rule set to use (conway, high_life, etc.)",
    )
    parser.add_argument("--width", "-W", type=int, default=50, help="Board width")
    parser.add_argument("--height", "-H", type=int, default=50, help="Board height")
    parser.add_argument(
        "--boundary",
        "-b",
        default="fixed",
        choices=["fixed", "toroidal"],
        help="Boundary condition",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.pattern:
        batch_mode(
            pattern_file=args.pattern,
            generations=args.generations,
            rule=args.rule,
            width=args.width,
            height=args.height,
            boundary=args.boundary,
        )
    else:
        print("No pattern file specified. Running in interactive mode...")
        interactive_mode()


if __name__ == "__main__":
    main()