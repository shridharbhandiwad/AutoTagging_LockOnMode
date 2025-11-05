"""
Binary parser for C-style struct definitions with endianness support.
Supports #pragma pack and various data types.
"""
import struct
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class Endianness(Enum):
    LITTLE = '<'
    BIG = '>'
    NATIVE = '='


@dataclass
class FieldDef:
    """Definition of a struct field"""
    name: str
    type: str
    count: int = 1  # Array size
    offset: Optional[int] = None


class StructParser:
    """Parse binary files based on C-style struct definitions"""
    
    TYPE_MAP = {
        'uint8_t': 'B',
        'int8_t': 'b',
        'uint16_t': 'H',
        'int16_t': 'h',
        'uint32_t': 'I',
        'int32_t': 'i',
        'uint64_t': 'Q',
        'int64_t': 'q',
        'float': 'f',
        'double': 'd',
        'char': 'c',
    }
    
    def __init__(self, struct_def: List[FieldDef], 
                 endianness: Endianness = Endianness.LITTLE,
                 packed: bool = False):
        """
        Initialize parser with struct definition.
        
        Args:
            struct_def: List of field definitions
            endianness: Byte order (little/big/native)
            packed: Whether struct is packed (#pragma pack)
        """
        self.struct_def = struct_def
        self.endianness = endianness
        self.packed = packed
        self.format_string = self._build_format_string()
        self.struct_size = struct.calcsize(self.format_string)
        
    def _build_format_string(self) -> str:
        """Build struct format string from field definitions"""
        fmt = self.endianness.value
        
        for field in self.struct_def:
            if field.type not in self.TYPE_MAP:
                raise ValueError(f"Unknown type: {field.type}")
            
            type_char = self.TYPE_MAP[field.type]
            if field.count > 1:
                fmt += f"{field.count}{type_char}"
            else:
                fmt += type_char
        
        return fmt
    
    def parse_record(self, data: bytes, offset: int = 0) -> Dict[str, Any]:
        """
        Parse a single record from binary data.
        
        Args:
            data: Binary data buffer
            offset: Starting offset in buffer
            
        Returns:
            Dictionary with field names and values
        """
        record_data = data[offset:offset + self.struct_size]
        if len(record_data) < self.struct_size:
            raise ValueError(f"Insufficient data: expected {self.struct_size}, got {len(record_data)}")
        
        values = struct.unpack(self.format_string, record_data)
        
        # Map values to field names
        result = {}
        value_idx = 0
        
        for field in self.struct_def:
            if field.count > 1:
                # Array field
                result[field.name] = list(values[value_idx:value_idx + field.count])
                value_idx += field.count
            else:
                # Single value
                result[field.name] = values[value_idx]
                value_idx += 1
        
        return result
    
    def parse_file(self, filepath: str, max_records: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Parse entire binary file.
        
        Args:
            filepath: Path to binary file
            max_records: Maximum number of records to parse (None = all)
            
        Returns:
            List of parsed records
        """
        records = []
        
        with open(filepath, 'rb') as f:
            data = f.read()
        
        offset = 0
        record_count = 0
        
        while offset + self.struct_size <= len(data):
            if max_records and record_count >= max_records:
                break
            
            try:
                record = self.parse_record(data, offset)
                records.append(record)
                offset += self.struct_size
                record_count += 1
            except Exception as e:
                print(f"Error parsing record at offset {offset}: {e}")
                break
        
        return records


class TrackRecordParser(StructParser):
    """Parser for airborne track records"""
    
    @staticmethod
    def get_standard_track_struct() -> List[FieldDef]:
        """Get standard track measurement structure"""
        return [
            FieldDef("track_id", "uint32_t"),
            FieldDef("timestamp", "uint64_t"),
            FieldDef("range", "float"),
            FieldDef("azimuth", "float"),
            FieldDef("elevation", "float"),
            FieldDef("range_rate", "float"),
            FieldDef("snr", "float"),
            FieldDef("rcs", "float"),
            FieldDef("doppler", "float"),
            FieldDef("position", "float", 3),  # x, y, z
            FieldDef("velocity", "float", 3),  # vx, vy, vz
            FieldDef("acceleration", "float", 3),  # ax, ay, az
            FieldDef("pos_error", "float", 3),
            FieldDef("vel_error", "float", 3),
            FieldDef("valid_flags", "uint32_t"),
        ]
    
    @staticmethod
    def from_header_file(header_path: str, 
                        endianness: Endianness = Endianness.LITTLE) -> 'TrackRecordParser':
        """
        Create parser from C header file (simplified parser).
        
        Args:
            header_path: Path to .h file with struct definition
            endianness: Byte order
            
        Returns:
            Configured parser
        """
        # For demo purposes, use standard struct
        # In production, implement full C header parser
        struct_def = TrackRecordParser.get_standard_track_struct()
        return TrackRecordParser(struct_def, endianness, packed=True)


def parse_binary_file(filepath: str, 
                     struct_def: Optional[List[FieldDef]] = None,
                     endianness: Endianness = Endianness.LITTLE) -> List[Dict[str, Any]]:
    """
    Convenience function to parse binary file.
    
    Args:
        filepath: Path to binary file
        struct_def: Struct definition (uses standard if None)
        endianness: Byte order
        
    Returns:
        List of parsed records
    """
    if struct_def is None:
        struct_def = TrackRecordParser.get_standard_track_struct()
    
    parser = TrackRecordParser(struct_def, endianness, packed=True)
    return parser.parse_file(filepath)
