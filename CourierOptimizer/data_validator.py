# data_validator.py
"""
Data validation module for CourierOptimizer package.
Handles CSV input validation, regex checks, and custom error handling.
"""

import csv
import re
import logging
from typing import List, Tuple, Dict, Any
from models import Delivery, InvalidDataError, Priority  # <-- import Priority enum

# Configure logging (fine for running this file directly)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidator:
    """Validates CSV input data and separates valid/invalid rows."""
    
    # Regex patterns for validation
    PRIORITY_PATTERN = re.compile(r'^(High|Medium|Low)$', re.IGNORECASE)
    CUSTOMER_NAME_PATTERN = re.compile(r'^[\w\s\-\.]+$')  # Alphanumeric, spaces, hyphens, dots
    NUMERIC_PATTERN = re.compile(r'^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$')  # accept sci-notation too
    
    # Expected CSV headers
    EXPECTED_HEADERS = ['customer', 'latitude', 'longitude', 'priority', 'weight_kg']
    
    # Geographic boundaries (Oslo area constraints)
    MIN_LATITUDE = 59.0
    MAX_LATITUDE = 60.5
    MIN_LONGITUDE = 10.0
    MAX_LONGITUDE = 11.5
    MAX_WEIGHT_KG = 50.0  # Reasonable limit for courier service

    @classmethod
    def validate_input(cls, filepath: str) -> Tuple[List[Delivery], List[Dict[str, Any]]]:
        """
        Main validation function that processes CSV input.
        
        Args:
            filepath: Path to the input CSV file
            
        Returns:
            Tuple of (valid_deliveries, rejected_rows)
        """
        valid_deliveries = []
        rejected_rows = []
        
        try:
            with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
                # Use DictReader to access columns by name
                reader = csv.DictReader(csvfile)
                
                # Validate CSV headers
                cls._validate_headers(reader.fieldnames)
                
                for row_num, raw_row in enumerate(reader, start=2):  # start=2 for header row
                    # skip fully empty lines
                    if all((v is None) or (str(v).strip() == '') for v in raw_row.values()):
                        continue

                    try:
                        # Validate the current row
                        validated_data = cls._validate_row(raw_row, row_num)

                        # Convert priority string -> Priority enum (matches your models.py)
                        priority_str = validated_data['priority']  # e.g., "High"
                        try:
                            priority_enum = Priority(priority_str)           # by value
                        except ValueError:
                            priority_enum = Priority[priority_str.upper()]   # by name

                        # Create Delivery object if validation passes
                        delivery = Delivery(
                            customer=validated_data['customer'],
                            latitude=validated_data['latitude'],
                            longitude=validated_data['longitude'],
                            priority=priority_enum,                       # <-- Enum, not string
                            weight_kg=validated_data['weight_kg']
                        )
                        
                        valid_deliveries.append(delivery)
                        logger.info(f"Row {row_num}: valid delivery for {validated_data['customer']}")
                        
                    except InvalidDataError as e:
                        # Add rejection reason to the row data
                        rejected_row = raw_row.copy()
                        rejected_row['rejection_reason'] = str(e)
                        rejected_row['row_number'] = row_num
                        rejected_rows.append(rejected_row)
                        logger.warning(f"Row {row_num} rejected: {e}")
                        
        except FileNotFoundError:
            error_msg = f"Input file not found: {filepath}"
            logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Error reading CSV file: {str(e)}"
            logger.exception(error_msg)
            raise
        
        # Write rejected rows to file
        if rejected_rows:
            cls._write_rejected_rows(rejected_rows)
        
        logger.info(f"Validation complete: {len(valid_deliveries)} valid, {len(rejected_rows)} rejected")
        return valid_deliveries, rejected_rows

    @classmethod
    def _validate_headers(cls, fieldnames: List[str]) -> None:
        """Validate that CSV has expected headers."""
        if not fieldnames:
            raise InvalidDataError("CSV file is empty or has no headers")
        
        # Strip BOM and normalize case/spaces
        normalized = [fn.lower().lstrip('\ufeff').strip() for fn in fieldnames]
        missing_headers = [h for h in cls.EXPECTED_HEADERS if h not in normalized]
        if missing_headers:
            raise InvalidDataError(f"Missing required headers: {missing_headers}")

    @classmethod
    def _validate_row(cls, raw_row: Dict[str, str], row_num: int) -> Dict[str, Any]:
        """
        Validate a single row of data.
        
        Args:
            raw_row: Dictionary of row data
            row_num: Row number for error messages
            
        Returns:
            Dictionary of validated and converted data
        """
        validated_data = {}
        
        # Check A: Data Presence & Basic Format
        cls._check_data_presence(raw_row, row_num)
        
        # Check B: Customer Name Validation
        customer = raw_row['customer'].strip()
        if not customer:
            raise InvalidDataError("Customer name cannot be empty", raw_row)
        if not cls.CUSTOMER_NAME_PATTERN.match(customer):
            raise InvalidDataError(f"Invalid customer name format: {customer}", raw_row)
        validated_data['customer'] = customer
        
        # Check C: Numeric Field Validation and Conversion
        validated_data['latitude'] = cls._validate_numeric_field(
            raw_row['latitude'], 'latitude', row_num, raw_row
        )
        validated_data['longitude'] = cls._validate_numeric_field(
            raw_row['longitude'], 'longitude', row_num, raw_row
        )
        validated_data['weight_kg'] = cls._validate_numeric_field(
            raw_row['weight_kg'], 'weight_kg', row_num, raw_row
        )
        
        # Check D: Geographic Boundaries
        cls._validate_geographic_bounds(
            validated_data['latitude'], validated_data['longitude'], row_num, raw_row
        )
        
        # Check E: Weight Constraints
        if validated_data['weight_kg'] < 0:
            raise InvalidDataError("Weight cannot be negative", raw_row)
        if validated_data['weight_kg'] > cls.MAX_WEIGHT_KG:
            raise InvalidDataError(
                f"Weight {validated_data['weight_kg']}kg exceeds maximum {cls.MAX_WEIGHT_KG}kg", 
                raw_row
            )
        
        # Check F: Priority Validation with Regex
        priority = raw_row['priority'].strip()
        if not cls.PRIORITY_PATTERN.match(priority):
            raise InvalidDataError(
                f"Invalid priority: {priority}. Must be High, Medium, or Low", 
                raw_row
            )
        validated_data['priority'] = priority.capitalize()  # Standardize case
        
        return validated_data

    @classmethod
    def _check_data_presence(cls, raw_row: Dict[str, str], row_num: int) -> None:
        """Check that all required fields are present and non-empty."""
        for header in cls.EXPECTED_HEADERS:
            if header not in raw_row:
                raise InvalidDataError(f"Missing column: {header}", raw_row)
            if not raw_row[header] or str(raw_row[header]).strip() == '':
                raise InvalidDataError(f"Empty value for {header}", raw_row)

    @staticmethod
    def _strip_inline_comment(s: str) -> str:
        """Remove inline comments of the form ' # ...' commonly used in the demo CSV."""
        return s.split(' #', 1)[0].strip()

    @classmethod
    def _validate_numeric_field(cls, value: str, field_name: str, row_num: int, 
                               raw_row: Dict[str, str]) -> float:
        """Validate and convert numeric fields."""
        value = (value or '').strip()
        value = cls._strip_inline_comment(value)  # <-- accept inline comments
        value = value.replace(',', '.')           # optional: allow decimal commas
        
        # Check if value matches numeric pattern
        if not cls.NUMERIC_PATTERN.match(value):
            raise InvalidDataError(f"Invalid numeric format for {field_name}: {value!r}", raw_row)
        
        try:
            numeric_value = float(value)
            return numeric_value
        except ValueError:
            raise InvalidDataError(f"Could not convert {field_name} to number: {value!r}", raw_row)

    @classmethod
    def _validate_geographic_bounds(cls, lat: float, lon: float, row_num: int, 
                                   raw_row: Dict[str, str]) -> None:
        """Validate geographic coordinates are within reasonable bounds."""
        # Global bounds
        if not (-90 <= lat <= 90):
            raise InvalidDataError(f"Latitude {lat} outside valid range [-90, 90]", raw_row)
        if not (-180 <= lon <= 180):
            raise InvalidDataError(f"Longitude {lon} outside valid range [-180, 180]", raw_row)
        
        # Oslo area bounds (optional but helpful)
        if not (cls.MIN_LATITUDE <= lat <= cls.MAX_LATITUDE):
            raise InvalidDataError(
                f"Latitude {lat} outside Oslo area range [{cls.MIN_LATITUDE}, {cls.MAX_LATITUDE}]", 
                raw_row
            )
        if not (cls.MIN_LONGITUDE <= lon <= cls.MAX_LONGITUDE):
            raise InvalidDataError(
                f"Longitude {lon} outside Oslo area range [{cls.MIN_LONGITUDE}, {cls.MAX_LONGITUDE}]", 
                raw_row
            )

    @classmethod
    def _write_rejected_rows(cls, rejected_rows: List[Dict[str, Any]]) -> None:
        """Write rejected rows to rejected.csv with rejection reasons."""
        try:
            with open('rejected.csv', 'w', newline='', encoding='utf-8') as csvfile:
                if rejected_rows:
                    # Ensure stable header order
                    fieldnames = cls.EXPECTED_HEADERS + ['row_number', 'rejection_reason']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                    for r in rejected_rows:
                        out = {k: r.get(k, '') for k in fieldnames}
                        writer.writerow(out)
                    
            logger.info(f"Written {len(rejected_rows)} rejected rows to rejected.csv")
        except Exception as e:
            logger.error(f"Failed to write rejected.csv: {str(e)}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Test with a sample CSV (auto-creates a test file)
    try:
        # Create a simple test file
        test_data = """customer,latitude,longitude,priority,weight_kg
Customer A,59.9139,10.7522,High,2.5
Customer B,59.911,10.750,Medium,15.0
Invalid Customer,91.0,10.750,High,5.0  # Invalid latitude
Customer C,59.915,10.753,InvalidPriority,3.0  # Invalid priority
Customer D,59.914,10.751,Low,-1.0  # Negative weight
,59.914,10.751,High,2.0  # Empty customer name
Customer E,59.916,10.754,High,60.0  # Excessive weight"""
        
        with open('test_input.csv', 'w', newline='', encoding='utf-8') as f:
            f.write(test_data)
        
        print("Testing data validator...")
        valid_deliveries, rejected_rows = DataValidator.validate_input('test_input.csv')
        
        print(f"\n✅ Valid deliveries ({len(valid_deliveries)}):")
        for delivery in valid_deliveries:
            # Safe print even if someone changes Delivery later
            pr = getattr(delivery.priority, 'value', delivery.priority)
            print(f"  - {delivery.customer} @ ({delivery.latitude}, {delivery.longitude}) "
                  f"[priority={pr}, weight_kg={delivery.weight_kg}]")
        
        print(f"\n❌ Rejected rows ({len(rejected_rows)}):")
        for rejected in rejected_rows:
            print(f"  - Row {rejected.get('row_number')}: {rejected.get('rejection_reason')}")
            
    except Exception as e:
        print(f"Test failed: {e}")