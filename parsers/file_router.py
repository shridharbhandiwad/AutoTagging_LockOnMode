"""
File router that detects file type and routes to appropriate parser.
"""
import os
from typing import Union, List, Dict, Any
import pandas as pd
from parsers.binary.struct_parser import parse_binary_file, Endianness
from parsers.text.text_parser import parse_text_file, TextParser


class FileType:
    BINARY = "binary"
    TEXT = "text"
    UNKNOWN = "unknown"


class FileRouter:
    """Route files to appropriate parser based on type detection"""
    
    @staticmethod
    def detect_file_type(filepath: str) -> str:
        """
        Detect whether file is binary or text.
        
        Args:
            filepath: Path to file
            
        Returns:
            File type (binary/text/unknown)
        """
        # Check file extension first
        ext = os.path.splitext(filepath)[1].lower()
        
        text_extensions = ['.txt', '.csv', '.json', '.jsonl', '.log', '.dat']
        binary_extensions = ['.bin', '.raw', '.dump']
        
        if ext in text_extensions:
            return FileType.TEXT
        elif ext in binary_extensions:
            return FileType.BINARY
        
        # Try to read first bytes to detect
        try:
            with open(filepath, 'rb') as f:
                sample = f.read(512)
            
            # Check if contains mostly printable ASCII
            text_chars = sum(1 for b in sample if 32 <= b <= 126 or b in [9, 10, 13])
            text_ratio = text_chars / len(sample) if sample else 0
            
            if text_ratio > 0.8:
                return FileType.TEXT
            else:
                return FileType.BINARY
        except:
            return FileType.UNKNOWN
    
    @staticmethod
    def parse_file(filepath: str, 
                   file_type: str = None,
                   endianness: Endianness = Endianness.LITTLE) -> pd.DataFrame:
        """
        Parse file with automatic type detection.
        
        Args:
            filepath: Path to file
            file_type: Force specific type (auto-detect if None)
            endianness: Byte order for binary files
            
        Returns:
            DataFrame with parsed data
        """
        if file_type is None:
            file_type = FileRouter.detect_file_type(filepath)
        
        if file_type == FileType.BINARY:
            records = parse_binary_file(filepath, endianness=endianness)
            return pd.DataFrame(records)
        elif file_type == FileType.TEXT:
            return parse_text_file(filepath)
        else:
            raise ValueError(f"Unknown file type for: {filepath}")
    
    @staticmethod
    def validate_track_data(df: pd.DataFrame) -> bool:
        """
        Validate that parsed data contains required track fields.
        
        Args:
            df: Parsed dataframe
            
        Returns:
            True if valid track data
        """
        required_fields = ['track_id', 'timestamp']
        return all(field in df.columns for field in required_fields)
