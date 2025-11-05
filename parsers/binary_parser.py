"""Binary file parser with C-style struct support."""

import struct
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import construct as cs


class Endianness(Enum):
    """Byte order enumeration."""
    LITTLE = 'little'
    BIG = 'big'
    NATIVE = 'native'


@dataclass
class FieldDefinition:
    """Definition of a struct field."""
    name: str
    type: str  # uint8_t, uint16_t, int32_t, float, double
    count: int = 1  # For arrays
    
    def get_format_char(self) -> str:
        """Get struct format character."""
        type_map = {
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
        return type_map.get(self.type, 'B')
    
    def get_size(self) -> int:
        """Get field size in bytes."""
        size_map = {
            'uint8_t': 1, 'int8_t': 1,
            'uint16_t': 2, 'int16_t': 2,
            'uint32_t': 4, 'int32_t': 4,
            'uint64_t': 8, 'int64_t': 8,
            'float': 4, 'double': 8,
            'char': 1,
        }
        return size_map.get(self.type, 1) * self.count


@dataclass
class StructDefinition:
    """Definition of a C-style struct."""
    name: str
    fields: List[FieldDefinition]
    packed: bool = False
    endianness: Endianness = Endianness.LITTLE
    
    def get_format_string(self) -> str:
        """Get struct format string."""
        endian_map = {
            Endianness.LITTLE: '<',
            Endianness.BIG: '>',
            Endianness.NATIVE: '=',
        }
        prefix = endian_map[self.endianness]
        
        fmt = prefix
        for field in self.fields:
            if field.count > 1:
                fmt += str(field.count)
            fmt += field.get_format_char()
        
        return fmt
    
    def get_size(self) -> int:
        """Get total struct size in bytes."""
        if self.packed:
            return sum(field.get_size() for field in self.fields)
        else:
            # Simple alignment: align each field to its size
            size = 0
            for field in self.fields:
                field_size = field.get_size()
                # Align to field size (simplified)
                alignment = min(field_size // field.count if field.count > 0 else 1, 8)
                if size % alignment != 0:
                    size += alignment - (size % alignment)
                size += field_size
            return size


class BinaryParser:
    """Parser for binary files using struct definitions."""
    
    def __init__(self, struct_def: StructDefinition):
        """Initialize parser with struct definition."""
        self.struct_def = struct_def
        self.format_string = struct_def.get_format_string()
        self.record_size = struct.calcsize(self.format_string)
    
    def parse_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Parse binary file into list of records."""
        records = []
        
        with open(filepath, 'rb') as f:
            data = f.read()
        
        num_records = len(data) // self.record_size
        
        for i in range(num_records):
            offset = i * self.record_size
            record_data = data[offset:offset + self.record_size]
            
            if len(record_data) < self.record_size:
                break
            
            try:
                values = struct.unpack(self.format_string, record_data)
                record = self._values_to_dict(values)
                records.append(record)
            except struct.error as e:
                print(f"Warning: Error parsing record {i}: {e}")
                continue
        
        return records
    
    def _values_to_dict(self, values: Tuple) -> Dict[str, Any]:
        """Convert unpacked values to dictionary."""
        record = {}
        value_idx = 0
        
        for field in self.struct_def.fields:
            if field.count == 1:
                record[field.name] = values[value_idx]
                value_idx += 1
            else:
                # Array field
                record[field.name] = list(values[value_idx:value_idx + field.count])
                value_idx += field.count
        
        return record
    
    def parse_to_dataframe(self, filepath: str):
        """Parse binary file to pandas DataFrame."""
        import pandas as pd
        
        records = self.parse_file(filepath)
        if not records:
            return pd.DataFrame()
        
        # Flatten array fields
        flattened_records = []
        for record in records:
            flat_record = {}
            for key, value in record.items():
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        flat_record[f"{key}_{i}"] = v
                else:
                    flat_record[key] = value
            flattened_records.append(flat_record)
        
        return pd.DataFrame(flattened_records)


class TrackRecordParser(BinaryParser):
    """Specialized parser for track records."""
    
    @staticmethod
    def create_default_struct() -> StructDefinition:
        """Create default track record structure."""
        fields = [
            FieldDefinition('timestamp', 'uint64_t'),
            FieldDefinition('track_id', 'uint32_t'),
            FieldDefinition('range', 'float'),
            FieldDefinition('azimuth', 'float'),
            FieldDefinition('elevation', 'float'),
            FieldDefinition('range_rate', 'float'),
            FieldDefinition('snr', 'float'),
            FieldDefinition('rcs', 'float'),
            FieldDefinition('doppler', 'float'),
            FieldDefinition('position', 'float', 3),  # x, y, z
            FieldDefinition('velocity', 'float', 3),  # vx, vy, vz
            FieldDefinition('acceleration', 'float', 3),  # ax, ay, az
            FieldDefinition('pos_error', 'float', 3),
            FieldDefinition('vel_error', 'float', 3),
            FieldDefinition('measurement_valid', 'uint8_t'),
        ]
        
        return StructDefinition(
            name='TrackRecord',
            fields=fields,
            packed=True,
            endianness=Endianness.LITTLE
        )
    
    def __init__(self, struct_def: Optional[StructDefinition] = None):
        """Initialize with default or custom struct."""
        if struct_def is None:
            struct_def = self.create_default_struct()
        super().__init__(struct_def)


class MeasurementRecordParser(BinaryParser):
    """Specialized parser for measurement records."""
    
    @staticmethod
    def create_default_struct() -> StructDefinition:
        """Create default measurement record structure."""
        fields = [
            FieldDefinition('timestamp', 'uint64_t'),
            FieldDefinition('measurement_id', 'uint32_t'),
            FieldDefinition('range', 'float'),
            FieldDefinition('azimuth', 'float'),
            FieldDefinition('elevation', 'float'),
            FieldDefinition('range_rate', 'float'),
            FieldDefinition('snr', 'float'),
            FieldDefinition('rcs', 'float'),
            FieldDefinition('doppler', 'float'),
            FieldDefinition('i_sample', 'float'),
            FieldDefinition('q_sample', 'float'),
            FieldDefinition('sp_value', 'float'),
        ]
        
        return StructDefinition(
            name='MeasurementRecord',
            fields=fields,
            packed=True,
            endianness=Endianness.LITTLE
        )
    
    def __init__(self, struct_def: Optional[StructDefinition] = None):
        """Initialize with default or custom struct."""
        if struct_def is None:
            struct_def = self.create_default_struct()
        super().__init__(struct_def)
