"""File type detection and routing."""

from pathlib import Path
from typing import Tuple, Optional


class FileDetector:
    """Detects file type and routes to appropriate parser."""
    
    @staticmethod
    def detect_file_type(filepath: str) -> Tuple[str, Optional[str]]:
        """
        Detect if file is binary or text.
        
        Returns:
            Tuple of (file_type, subtype) where:
                file_type: 'binary' or 'text'
                subtype: 'csv', 'json', 'whitespace', etc. for text; None for binary
        """
        path = Path(filepath)
        
        # Check extension first
        ext = path.suffix.lower()
        text_extensions = {'.csv', '.txt', '.log', '.json', '.jsonl', '.tsv'}
        binary_extensions = {'.bin', '.dat', '.raw'}
        
        if ext in text_extensions:
            return 'text', None
        elif ext in binary_extensions:
            return 'binary', None
        
        # Read first few bytes to determine
        try:
            with open(filepath, 'rb') as f:
                header = f.read(512)
            
            # Check if mostly printable ASCII
            printable_ratio = sum(32 <= b <= 126 or b in (9, 10, 13) for b in header) / len(header)
            
            if printable_ratio > 0.8:
                return 'text', None
            else:
                return 'binary', None
        except Exception:
            return 'unknown', None
    
    @staticmethod
    def get_parser(filepath: str):
        """Get appropriate parser for file."""
        from .binary_parser import TrackRecordParser
        from .text_parser import TextParser
        
        file_type, _ = FileDetector.detect_file_type(filepath)
        
        if file_type == 'binary':
            return TrackRecordParser()
        elif file_type == 'text':
            return TextParser()
        else:
            raise ValueError(f"Cannot determine parser for file: {filepath}")
