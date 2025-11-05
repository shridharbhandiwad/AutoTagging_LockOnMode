"""
Text file parser supporting CSV, whitespace-delimited, and JSON-lines formats.
Auto-detects format from file content.
"""
import csv
import json
from typing import List, Dict, Any, Optional
from enum import Enum
import pandas as pd


class TextFormat(Enum):
    CSV = "csv"
    WHITESPACE = "whitespace"
    JSON_LINES = "jsonl"
    UNKNOWN = "unknown"


class TextParser:
    """Parse text files in various formats"""
    
    @staticmethod
    def detect_format(filepath: str) -> TextFormat:
        """
        Auto-detect text file format from first few lines.
        
        Args:
            filepath: Path to text file
            
        Returns:
            Detected format
        """
        with open(filepath, 'r') as f:
            # Read first non-empty line
            first_line = ""
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    first_line = stripped
                    break
            
            if not first_line:
                return TextFormat.UNKNOWN
            
            # Check for JSON
            if first_line.startswith('{') and first_line.endswith('}'):
                try:
                    json.loads(first_line)
                    return TextFormat.JSON_LINES
                except:
                    pass
            
            # Check for CSV (commas)
            if ',' in first_line:
                return TextFormat.CSV
            
            # Check for whitespace-delimited
            if len(first_line.split()) > 1:
                return TextFormat.WHITESPACE
            
            return TextFormat.UNKNOWN
    
    @staticmethod
    def parse_csv(filepath: str, delimiter: str = ',') -> pd.DataFrame:
        """
        Parse CSV file.
        
        Args:
            filepath: Path to CSV file
            delimiter: Field delimiter
            
        Returns:
            DataFrame with parsed data
        """
        return pd.read_csv(filepath, delimiter=delimiter, comment='#')
    
    @staticmethod
    def parse_whitespace(filepath: str) -> pd.DataFrame:
        """
        Parse whitespace-delimited file.
        
        Args:
            filepath: Path to file
            
        Returns:
            DataFrame with parsed data
        """
        return pd.read_csv(filepath, delim_whitespace=True, comment='#')
    
    @staticmethod
    def parse_jsonlines(filepath: str) -> pd.DataFrame:
        """
        Parse JSON-lines file.
        
        Args:
            filepath: Path to JSONL file
            
        Returns:
            DataFrame with parsed data
        """
        records = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line: {line[:50]}... Error: {e}")
        
        return pd.DataFrame(records)
    
    @staticmethod
    def parse_file(filepath: str, format: Optional[TextFormat] = None) -> pd.DataFrame:
        """
        Parse text file with auto-detection or specified format.
        
        Args:
            filepath: Path to file
            format: Format to use (auto-detect if None)
            
        Returns:
            DataFrame with parsed data
        """
        if format is None:
            format = TextParser.detect_format(filepath)
        
        if format == TextFormat.CSV:
            return TextParser.parse_csv(filepath)
        elif format == TextFormat.WHITESPACE:
            return TextParser.parse_whitespace(filepath)
        elif format == TextFormat.JSON_LINES:
            return TextParser.parse_jsonlines(filepath)
        else:
            raise ValueError(f"Unknown or unsupported format: {format}")
    
    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to standard format.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with normalized column names
        """
        # Standard column name mappings
        column_map = {
            'trackid': 'track_id',
            'track': 'track_id',
            'id': 'track_id',
            'time': 'timestamp',
            't': 'timestamp',
            'r': 'range',
            'az': 'azimuth',
            'el': 'elevation',
            'rr': 'range_rate',
            'range_dot': 'range_rate',
            'x': 'pos_x',
            'y': 'pos_y',
            'z': 'pos_z',
            'vx': 'vel_x',
            'vy': 'vel_y',
            'vz': 'vel_z',
        }
        
        # Lowercase all column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Apply mappings
        df = df.rename(columns=column_map)
        
        return df


def parse_text_file(filepath: str) -> pd.DataFrame:
    """
    Convenience function to parse text file with auto-detection.
    
    Args:
        filepath: Path to text file
        
    Returns:
        DataFrame with normalized data
    """
    df = TextParser.parse_file(filepath)
    df = TextParser.normalize_columns(df)
    return df
