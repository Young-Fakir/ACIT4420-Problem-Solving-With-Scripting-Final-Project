"""
pattern_parser.py - Pattern File Parser for Conway's Game of Life

This module handles reading initial patterns from configuration files using
regular expressions for parsing various pattern formats.
"""

import re
import logging
from typing import List, Tuple, Dict, Any
from pathlib import Path

# Module logger
_logger = logging.getLogger(__name__)

# Custom exceptions
class PatternParseError(Exception):
    """Raised when pattern parsing fails"""
    pass

class PatternFileError(Exception):
    """Raised when pattern file operations fail"""
    pass


class PatternParser:
    """
    Parses various pattern file formats for Conway's Game of Life.
    
    Supported formats:
    - RLE (Run Length Encoding)
    - Plain text (with various symbol conventions)
    - Coordinate format
    - Life 1.06 format
    """
    
    # Regular expressions for different pattern formats
    RLE_TOKEN = re.compile(r'(\d*)([bo$!])', re.IGNORECASE)
    COORDINATE_PATTERN = re.compile(r'^\s*(-?\d+)\s*[,\s]\s*(-?\d+)\s*$')  # allow negatives
    LIFE_106_PATTERN = re.compile(r'^\s*(-?\d+)\s+(-?\d+)\s*$')            # allow negatives
    # For detection only (not parsing). Broad, but avoids treating comments-only as plain.
    SYMBOL_DETECT = re.compile(r'[.*x+o+#\-\s]', re.IGNORECASE)

    # Symbol mappings for plain text format (case-insensitive)
    # '#' is *not* included to avoid confusing comments with live cells.
    SYMBOL_MAP = {
        'X': True, 'O': True, '*': True,      # Live cells
        '.': False, ' ': False, '-': False     # Dead cells
    }
    
    def __init__(self):
        """Initialize the pattern parser with default settings."""
        self.supported_formats = ['rle', 'plain', 'coordinates', 'life106']
        _logger.info("Initialized PatternParser")
    
    def load_pattern(self, filename: str) -> List[Tuple[int, int]]:
        """
        Load a pattern from file and return list of (x, y) coordinates for live cells.
        
        Args:
            filename: Path to the pattern file
            
        Returns:
            List of (x, y) coordinates for live cells
            
        Raises:
            PatternFileError: If file cannot be read
            PatternParseError: If pattern format is invalid
        """
        file_path = Path(filename)
        if not file_path.exists():
            raise PatternFileError(f"Pattern file not found: {filename}")

        try:
            content = file_path.read_text(encoding='utf-8').strip()
        except UnicodeDecodeError as e:
            raise PatternFileError(f"Unable to read file {filename}: {e}") from e

        _logger.info("Loading pattern from: %s", filename)

        pattern_format = self._detect_format(content, filename)
        _logger.debug("Detected pattern format: %s", pattern_format)

        try:
            if pattern_format == 'rle':
                return self._parse_rle(content)
            elif pattern_format == 'plain':
                return self._parse_plain_text(content)
            elif pattern_format == 'coordinates':
                return self._parse_coordinates(content)
            elif pattern_format == 'life106':
                return self._parse_life106(content)
            else:
                raise PatternParseError(f"Unsupported pattern format in {filename}")
        except PatternParseError:
            raise
        except Exception as e:
            # Wrap unexpected parsing failures
            raise PatternParseError(f"Failed to load pattern from {filename}: {e}") from e
    
    def _detect_format(self, content: str, filename: str) -> str:
        """
        Auto-detect the pattern file format.
        """
        # Check file extension first
        ext = Path(filename).suffix.lower()
        if ext == '.rle':
            return 'rle'
        elif ext in ('.lif', '.life'):
            return 'life106'
        
        # Content-based detection
        lines = content.splitlines()
        first_line = lines[0].strip() if lines else ""

        # Life 1.06 header
        if first_line == '#Life 1.06':
            return 'life106'

        # RLE header (x = N, y = M, [rule = ...])
        if any(('x =' in ln.lower() and 'y =' in ln.lower()) for ln in lines):
            return 'rle'
        
        # Coordinate format (first non-comment line looks like "x,y" or "x y")
        for ln in lines:
            s = ln.strip()
            if not s or s.startswith('#'):
                continue
            if self.COORDINATE_PATTERN.match(s):
                return 'coordinates'
            break  # first non-comment line didn't match coordinate style

        # Plain text: contains symbol-ish characters
        if lines and self.SYMBOL_DETECT.search(lines[0]):
            return 'plain'
        
        _logger.warning("Could not detect pattern format, defaulting to plain text")
        return 'plain'
    
    def _parse_rle(self, content: str) -> List[Tuple[int, int]]:
        """
        Parse RLE (Run Length Encoding) format.

        Example:
            #N Glider
            x = 3, y = 3, rule = B3/S23
            bo$2bo$3o!
        """
        _logger.debug("Parsing RLE format")
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

        # Find the header line (x=..., y=...) then pattern lines follow
        pattern_lines: List[str] = []
        in_pattern = False

        for ln in lines:
            if ln.startswith('#'):
                continue
            low = ln.lower()
            if not in_pattern:
                # header detected; skip the header line itself
                if 'x =' in low and 'y =' in low:
                    in_pattern = True
                    continue
            else:
                pattern_lines.append(ln)

        if not pattern_lines:
            raise PatternParseError("RLE content missing pattern data after header")

        # Join RLE data, strip comments/whitespace and cut at '!'
        rle_str = ''.join(pattern_lines)
        excl_idx = rle_str.find('!')
        if excl_idx != -1:
            rle_str = rle_str[:excl_idx]

        x = y = 0
        live_cells: List[Tuple[int, int]] = []

        for count_str, code in self.RLE_TOKEN.findall(rle_str):
            count = int(count_str) if count_str else 1
            c = code.lower()
            if c == 'b':
                x += count
            elif c == 'o':
                # add 'count' live cells at (x..x+count-1, y)
                for i in range(count):
                    live_cells.append((x + i, y))
                x += count
            elif c == '$':
                y += count
                x = 0
            # '!' handled by truncation above

        _logger.info("Parsed %d live cells from RLE format", len(live_cells))
        return live_cells
    
    def _parse_plain_text(self, content: str) -> List[Tuple[int, int]]:
        """
        Parse plain text format with symbols for live/dead cells.

        Examples:
            .O.
            O.O
            .O.
        """
        _logger.debug("Parsing plain text format")
        # Keep non-empty, non-comment lines
        raw_lines = [line.rstrip('\n') for line in content.splitlines()]
        lines = [ln for ln in raw_lines if ln.strip() and not ln.lstrip().startswith('#')]

        live_cells: List[Tuple[int, int]] = []
        
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                is_live = self.SYMBOL_MAP.get(char.upper(), False)
                if is_live:
                    live_cells.append((x, y))
        
        _logger.info("Parsed %d live cells from plain text format", len(live_cells))
        return live_cells
    
    def _parse_coordinates(self, content: str) -> List[Tuple[int, int]]:
        """
        Parse coordinate format (one coordinate pair per line).

        Examples:
            1, 2
            3,4
            -5 6
        """
        _logger.debug("Parsing coordinate format")
        lines = content.splitlines()
        live_cells: List[Tuple[int, int]] = []
        
        for line_num, line in enumerate(lines, 1):
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            
            match = self.COORDINATE_PATTERN.match(s)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                live_cells.append((x, y))
            else:
                _logger.warning("Skipping invalid coordinate line %d: %s", line_num, line)
        
        _logger.info("Parsed %d live cells from coordinate format", len(live_cells))
        return live_cells
    
    def _parse_life106(self, content: str) -> List[Tuple[int, int]]:
        """
        Parse Life 1.06 format (only coordinates, no size info).

        Example:
            #Life 1.06
            1 2
            -3 4
        """
        _logger.debug("Parsing Life 1.06 format")
        lines = content.splitlines()
        live_cells: List[Tuple[int, int]] = []
        
        # Skip header if present
        start_index = 1 if lines and lines[0].strip() == '#Life 1.06' else 0
        
        for line_num, line in enumerate(lines[start_index:], start_index + 1):
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            
            match = self.LIFE_106_PATTERN.match(s)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                live_cells.append((x, y))
            else:
                _logger.warning("Skipping invalid Life 1.06 line %d: %s", line_num, line)
        
        _logger.info("Parsed %d live cells from Life 1.06 format", len(live_cells))
        return live_cells
    
    def save_pattern(self, live_cells: List[Tuple[int, int]], filename: str, 
                     format: str = 'plain') -> None:
        """
        Save live cells to a pattern file.
        
        Args:
            live_cells: List of (x, y) coordinates
            filename: Output file path
            format: Output format ('plain', 'coordinates', or 'life106')
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                if format == 'plain':
                    self._save_plain_text(f, live_cells)
                elif format == 'coordinates':
                    self._save_coordinates(f, live_cells)
                elif format == 'life106':
                    self._save_life106(f, live_cells)
                else:
                    raise PatternParseError(f"Unsupported output format: {format}")
            
            _logger.info("Saved %d live cells to %s in %s format", 
                         len(live_cells), filename, format)
        except OSError as e:
            raise PatternFileError(f"Failed to write pattern file {filename}: {e}") from e
    
    def _save_plain_text(self, file_obj, live_cells: List[Tuple[int, int]]) -> None:
        """Save pattern in plain text format."""
        if not live_cells:
            file_obj.write("# Empty pattern\n")
            return
        
        # Find bounds
        max_x = max(x for x, _ in live_cells)
        max_y = max(y for _, y in live_cells)
        min_x = min(x for x, _ in live_cells)
        min_y = min(y for _, y in live_cells)

        # Normalize to start at (0,0) for a compact export
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        
        grid = [['.' for _ in range(width)] for _ in range(height)]
        for x, y in live_cells:
            grid[y - min_y][x - min_x] = 'O'
        
        # Write to file
        for row in grid:
            file_obj.write(''.join(row) + '\n')
    
    def _save_coordinates(self, file_obj, live_cells: List[Tuple[int, int]]) -> None:
        """Save pattern in coordinate format."""
        file_obj.write("# Coordinate format pattern\n")
        for x, y in sorted(live_cells):
            file_obj.write(f"{x}, {y}\n")
    
    def _save_life106(self, file_obj, live_cells: List[Tuple[int, int]]) -> None:
        """Save pattern in Life 1.06 format."""
        file_obj.write("#Life 1.06\n")
        for x, y in sorted(live_cells):
            file_obj.write(f"{x} {y}\n")
    
    def get_format_info(self, filename: str) -> Dict[str, Any]:
        """
        Get information about a pattern file without loading it.
        """
        try:
            p = Path(filename)
            content = p.read_text(encoding='utf-8').strip()
            format_type = self._detect_format(content, filename)
            
            # Count lines and estimate size
            lines = [line for line in content.splitlines()]
            non_empty = [ln for ln in lines if ln.strip()]
            non_comment = [ln for ln in non_empty if not ln.lstrip().startswith('#')]
            
            return {
                'format': format_type,
                'total_lines': len(lines),
                'non_empty_lines': len(non_empty),
                'non_comment_lines': len(non_comment),
                'file_size': p.stat().st_size,
                'supported': format_type in self.supported_formats
            }
        except Exception as e:
            _logger.error("Failed to get format info for %s: %s", filename, e)
            return {'error': str(e)}


# Convenience functions
def load_pattern(filename: str) -> List[Tuple[int, int]]:
    parser = PatternParser()
    return parser.load_pattern(filename)


def save_pattern(live_cells: List[Tuple[int, int]], filename: str, 
                 format: str = 'plain') -> None:
    parser = PatternParser()
    parser.save_pattern(live_cells, filename, format)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        parser = PatternParser()
        
        # Test with sample patterns
        test_patterns = [
            # Plain text pattern
            "plain_test.txt",
            # Coordinate pattern  
            "coord_test.txt",
            # You can create these test files to verify functionality
        ]
        
        for pattern_file in test_patterns:
            if Path(pattern_file).exists():
                print(f"Loading {pattern_file}:")
                cells = parser.load_pattern(pattern_file)
                print(f"  Found {len(cells)} live cells")
                
                info = parser.get_format_info(pattern_file)
                print(f"  Format info: {info}")
                
                # Test saving in different format
                output_file = f"output_{Path(pattern_file).stem}.coords"
                parser.save_pattern(cells, output_file, 'coordinates')
                print(f"  Saved to {output_file}")
                
            else:
                print(f"Test file {pattern_file} not found, creating sample...")
                
                # Create sample plain text pattern (glider)
                sample_cells = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)]
                parser.save_pattern(sample_cells, "sample_glider.txt", 'plain')
                print("  Created sample_glider.txt")
        
        print("\nAvailable pattern formats:", parser.supported_formats)
        
    except Exception as e:
        print(f"Error during testing: {e}")