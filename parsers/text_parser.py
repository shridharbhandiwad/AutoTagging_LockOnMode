"""Text file parser for CSV, whitespace-delimited, and JSON-lines formats."""

import pandas as pd
import json
from typing import Optional, List, Dict, Any
from pathlib import Path


class TextParser:
    """Parser for text-based track data files."""
    
    def __init__(self):
        """Initialize text parser."""
        self.detected_format = None
    
    def detect_format(self, filepath: str) -> str:
        """Auto-detect file format from first lines."""
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            
            # Check for JSON lines
            if first_line.startswith('{'):
                try:
                    json.loads(first_line)
                    return 'jsonl'
                except:
                    pass
            
            # Check for CSV (comma-separated)
            if ',' in first_line:
                return 'csv'
            
            # Check for TSV or whitespace-delimited
            if '\t' in first_line:
                return 'tsv'
            
            # Default to whitespace-delimited
            return 'whitespace'
    
    def parse_file(self, filepath: str, format: Optional[str] = None) -> pd.DataFrame:
        """Parse text file to DataFrame."""
        if format is None:
            format = self.detect_format(filepath)
        
        self.detected_format = format
        
        if format == 'csv':
            return pd.read_csv(filepath)
        elif format == 'tsv':
            return pd.read_csv(filepath, sep='\t')
        elif format == 'whitespace':
            return pd.read_csv(filepath, delim_whitespace=True)
        elif format == 'jsonl':
            return self._parse_jsonlines(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _parse_jsonlines(self, filepath: str) -> pd.DataFrame:
        """Parse JSON-lines format."""
        records = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        return pd.DataFrame(records)
    
    def parse_to_tracks(self, filepath: str, format: Optional[str] = None) -> Dict[int, pd.DataFrame]:
        """Parse file and group by track_id."""
        df = self.parse_file(filepath, format)
        
        # Try to find track_id column
        track_id_col = None
        for col in ['track_id', 'trackid', 'id', 'TrackID']:
            if col in df.columns:
                track_id_col = col
                break
        
        if track_id_col is None:
            # Assume single track
            return {0: df}
        
        # Group by track_id
        tracks = {}
        for track_id, group in df.groupby(track_id_col):
            tracks[int(track_id)] = group.reset_index(drop=True)
        
        return tracks
    
    @staticmethod
    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to expected format."""
        column_mapping = {
            'trackid': 'track_id',
            'TrackID': 'track_id',
            'id': 'track_id',
            'time': 'timestamp',
            'Time': 'timestamp',
            'rng': 'range',
            'Range': 'range',
            'az': 'azimuth',
            'Azimuth': 'azimuth',
            'el': 'elevation',
            'Elevation': 'elevation',
            'range_rt': 'range_rate',
            'RangeRate': 'range_rate',
            'SNR': 'snr',
            'RCS': 'rcs',
            'Doppler': 'doppler',
            'x': 'pos_x',
            'y': 'pos_y',
            'z': 'pos_z',
            'vx': 'vel_x',
            'vy': 'vel_y',
            'vz': 'vel_z',
        }
        
        df = df.copy()
        df.rename(columns=column_mapping, inplace=True)
        return df
