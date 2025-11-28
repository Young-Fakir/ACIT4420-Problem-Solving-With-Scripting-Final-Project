"""
board.py - Grid Management Module for Conway's Game of Life

This module defines the Board class which manages the game grid state,
including initialization, pattern loading, stepping, randomizing,
snapshot saving, and display.
"""

from typing import List, Tuple, TypedDict
import logging
import struct
import numpy as np

# Module logger
_logger = logging.getLogger(__name__)

# Custom exceptions
class GridDimensionError(Exception):
    """Raised when invalid grid dimensions are provided"""
    pass

class BoundaryError(Exception):
    """Raised when pattern coordinates exceed grid boundaries"""
    pass

class PatternLoadError(Exception):
    """Raised when pattern loading fails"""
    pass

class SnapshotSaveError(Exception):
    """Raised when saving a snapshot fails"""
    pass


class BoardStats(TypedDict):
    generation: int
    width: int
    height: int
    live_cells: int
    dead_cells: int
    density: float
    total_cells: int


class Board:
    """
    Manages the Game of Life grid state and operations.
    
    Attributes:
        width (int): Number of columns in the grid
        height (int): Number of rows in the grid
        grid (numpy.ndarray): 2D boolean array (False=dead, True=alive)
        generation (int): Current generation number
    """
    
    def __init__(self, width: int, height: int):
        """
        Initialize a new game board with the specified dimensions.
        """
        self._validate_dimensions(width, height)
        
        self.width = width
        self.height = height
        self.generation = 0
        # Use bool for clarity and memory efficiency
        self.grid = np.zeros((height, width), dtype=bool)

        # Instance logger (alias to module logger)
        self._logger = _logger
        
        self._logger.info("Initialized board: %dx%d", width, height)
    
    def _validate_dimensions(self, width: int, height: int) -> None:
        """Validate grid dimensions"""
        if not isinstance(width, int) or not isinstance(height, int):
            raise GridDimensionError("Grid dimensions must be integers")
        if width <= 0 or height <= 0:
            raise GridDimensionError("Grid dimensions must be positive integers")
    
    # -----------------------------
    # Pattern I/O
    # -----------------------------
    def load_pattern(self, pattern_file: str, pattern_parser) -> None:
        """
        Load a pattern from file and apply it to the grid (clearing the grid first).
        
        Args:
            pattern_file: Path to the pattern file
            pattern_parser: Instance with .load_pattern(file) -> List[Tuple[int,int]]
        """
        try:
            live_cells = pattern_parser.load_pattern(pattern_file)
        except FileNotFoundError as e:
            raise PatternLoadError(f"Pattern file not found: {pattern_file}") from e
        except (ValueError, PatternLoadError) as e:
            raise PatternLoadError(f"Failed to load pattern: {e}") from e

        # Apply and clear by default for file-loaded patterns
        self._apply_pattern(live_cells, clear=True)
        self._logger.info("Loaded pattern from %s: %d live cells", pattern_file, len(live_cells))
    
    def apply_cells(self, live_cells: List[Tuple[int, int]], *, clear: bool = False) -> None:
        """
        Public helper to apply a list of live cells to the grid.
        
        Args:
            live_cells: List of (x, y) coordinates for live cells
            clear: If True, clear the grid before applying
        """
        self._apply_pattern(live_cells, clear=clear)

    def _apply_pattern(self, live_cells: List[Tuple[int, int]], *, clear: bool = True) -> None:
        """
        Apply live cells to the grid at specified coordinates.
        """
        if clear:
            self.grid.fill(False)
        
        for x, y in live_cells:
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise BoundaryError(
                    f"Pattern coordinate ({x}, {y}) exceeds grid boundaries "
                    f"({self.width}x{self.height})"
                )
            self.grid[y, x] = True
    
    # -----------------------------
    # Cell ops
    # -----------------------------
    def set_cell(self, x: int, y: int, state: int) -> None:
        """
        Set the state of a specific cell.
        
        Args:
            x: X coordinate (column)
            y: Y coordinate (row)
            state: 0/False for dead, 1/True for alive
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise BoundaryError(f"Coordinate ({x}, {y}) is outside grid boundaries")
        if state not in (0, 1, False, True):
            raise ValueError("state must be 0/1 or bool")
        self.grid[y, x] = bool(state)
    
    def get_cell(self, x: int, y: int) -> int:
        """
        Get the state of a specific cell. Returns 0 or 1.
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise BoundaryError(f"Coordinate ({x}, {y}) is outside grid boundaries")
        return int(self.grid[y, x])
    
    def clear(self) -> None:
        """Clear the entire grid (set all cells to dead)"""
        self.grid.fill(False)
        self.generation = 0
        self._logger.info("Grid cleared")
    
    def resize(self, new_width: int, new_height: int) -> None:
        """
        Resize the grid while preserving as much of the current pattern as possible.
        """
        self._validate_dimensions(new_width, new_height)
        
        # Create new grid with same dtype
        new_grid = np.zeros((new_height, new_width), dtype=self.grid.dtype)
        
        # Copy overlapping region
        copy_width = min(self.width, new_width)
        copy_height = min(self.height, new_height)
        
        new_grid[:copy_height, :copy_width] = self.grid[:copy_height, :copy_width]
        
        self.grid = new_grid
        self.width = new_width
        self.height = new_height
        
        self._logger.info("Grid resized to %dx%d", new_width, new_height)

    # -----------------------------
    # Simulation (Game of Life)
    # -----------------------------
    def _neighbor_counts(self, wrap: bool) -> np.ndarray:
        """
        Compute neighbor counts for each cell.
        If wrap=True, edges wrap around (toroidal). If wrap=False, edges do not wrap.
        """
        H, W = self.height, self.width
        counts = np.zeros((H, W), dtype=np.uint8)

        if wrap:
            g = self.grid
            # Sum of 8 rolled neighbors
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    counts += np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        else:
            g = self.grid
            # Add neighbors by slicing without wrap
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    # Source slice
                    src_y_start = max(0, dy)
                    src_y_end = H + min(0, dy)
                    src_x_start = max(0, dx)
                    src_x_end = W + min(0, dx)

                    # Destination slice
                    dst_y_start = max(0, -dy)
                    dst_y_end = H - max(0, dy)
                    dst_x_start = max(0, -dx)
                    dst_x_end = W - max(0, dx)

                    if src_y_start < src_y_end and src_x_start < src_x_end:
                        counts[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += g[src_y_start:src_y_end, src_x_start:src_x_end]

        return counts

    def step(self, *, wrap: bool = False) -> None:
        """
        Advance the simulation by one generation using Conway's rules.
        
        Args:
            wrap: If True, grid edges wrap around (toroidal).
        """
        neighbors = self._neighbor_counts(wrap=wrap)

        alive = self.grid
        survive = alive & ((neighbors == 2) | (neighbors == 3))
        born = ~alive & (neighbors == 3)
        self.grid = (survive | born)
        self.generation += 1

    def step_n(self, n: int, *, wrap: bool = False) -> None:
        """
        Advance the simulation by n generations.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        for _ in range(n):
            self.step(wrap=wrap)

    def randomize(self, *, density: float = 0.25, seed: int | None = None, reset_generation: bool = True) -> None:
        """
        Fill the grid with random live cells.
        
        Args:
            density: Probability [0..1] that a given cell starts alive.
            seed: Optional RNG seed for reproducibility.
            reset_generation: If True, sets generation to 0 after randomizing.
        """
        if not (0.0 <= density <= 1.0):
            raise ValueError("density must be in [0, 1]")
        rng = np.random.default_rng(seed)
        self.grid = rng.random((self.height, self.width)) < density
        if reset_generation:
            self.generation = 0
        self._logger.info("Grid randomized (density=%.3f, seed=%s)", density, "None" if seed is None else str(seed))

    # -----------------------------
    # Snapshots
    # -----------------------------
    def save_snapshot(self, filename: str, format: str = "text") -> None:
        """
        Save the current grid state to a file.
        
        Args:
            filename: Output file path
            format: "text" for human-readable, "binary" for true binary
        """
        try:
            if format == "text":
                self._save_text_snapshot(filename)
            elif format == "binary":
                self._save_true_binary_snapshot(filename)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self._logger.info("Grid snapshot saved to %s (format: %s)", filename, format)
            
        except (IOError, OSError) as e:
            raise SnapshotSaveError(f"Failed to save snapshot: {e}") from e
    
    def _save_text_snapshot(self, filename: str) -> None:
        """Save grid as human-readable text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Game of Life Snapshot - Generation {self.generation}\n")
            f.write(f"# Dimensions: {self.width}x{self.height}\n")
            for row in self.grid:
                line = ''.join('█' if cell else ' ' for cell in row)
                f.write(line + '\n')
    
    def _save_true_binary_snapshot(self, filename: str) -> None:
        """
        Save grid as a true binary format:
        Header: width, height, generation as big-endian unsigned 64-bit integers
        Payload: packed bits row-major, little bit order within bytes.
        """
        with open(filename, "wb") as f:
            f.write(struct.pack(">QQQ", self.width, self.height, self.generation))
            packed = np.packbits(self.grid.astype(np.uint8), axis=None, bitorder="little")
            f.write(packed.tobytes())

    # -----------------------------
    # Display & stats
    # -----------------------------
    def display(self, symbols: Tuple[str, str] = (' ', '█')) -> None:
        """
        Display the current grid state in the console.
        
        Args:
            symbols: Tuple of (dead_symbol, live_symbol) for display
        """
        dead_char, live_char = symbols
        if not dead_char or not live_char:
            raise ValueError("symbols must be non-empty strings")
        
        print(f"Generation: {self.generation} | Grid: {self.width}x{self.height}")
        print('┌' + '─' * self.width + '┐')
        
        for row in self.grid:
            row_str = ''.join(live_char if cell else dead_char for cell in row)
            print('│' + row_str + '│')
        
        print('└' + '─' * self.width + '┘')
    
    def get_statistics(self) -> BoardStats:
        """
        Get statistics about the current grid state.
        """
        total = self.width * self.height
        live_cells = int(np.count_nonzero(self.grid))
        dead_cells = total - live_cells
        density = (live_cells / total) if total else 0.0
        
        return {
            'generation': self.generation,
            'width': self.width,
            'height': self.height,
            'live_cells': live_cells,
            'dead_cells': dead_cells,
            'density': density,
            'total_cells': total
        }
    
    def __str__(self) -> str:
        """String representation of the board"""
        stats = self.get_statistics()
        return (f"Board({self.width}x{self.height}, "
                f"generation={self.generation}, "
                f"live_cells={stats['live_cells']})")
    
    def __repr__(self) -> str:
        """Technical representation of the board"""
        return f"Board(width={self.width}, height={self.height}, generation={self.generation})"


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for demo purposes
    logging.basicConfig(level=logging.INFO)

    try:
        # Create a board
        board = Board(10, 10)
        print("Initial board:")
        board.display()
        
        # Initial manual setup
        board.set_cell(1, 1, 1)
        board.set_cell(2, 2, 1)
        board.set_cell(3, 3, 1)
        # Overlay a small block without clearing
        board.apply_cells([(0, 0), (0, 1), (1, 0), (1, 1)], clear=False)

        print("\nAfter initial setup:")
        board.display()

        # Step 3 times without wrapping
        board.step_n(3, wrap=False)
        print("\nAfter 3 steps (wrap=False):")
        board.display()

        # Stats + snapshots
        stats = board.get_statistics()
        print(f"\nStatistics: {stats}")
        board.save_snapshot("test_snapshot.txt", format="text")
        print("\nSnapshot saved to test_snapshot.txt (text)")
        board.save_snapshot("test_snapshot.bin", format="binary")
        print("Snapshot saved to test_snapshot.bin (binary)")

        # Randomize and show
        board.randomize(density=0.25, seed=42, reset_generation=True)
        print("\nRandomized grid:")
        board.display()

        # Single step with wrapping
        board.step(wrap=True)
        print("\nAfter 1 step (wrap=True):")
        board.display()

    except Exception as e:
        print(f"Error: {e}")