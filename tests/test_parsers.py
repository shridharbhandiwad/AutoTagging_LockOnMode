"""Tests for parsers."""

import pytest
import numpy as np
import tempfile
import struct
from pathlib import Path

from parsers.binary_parser import (BinaryParser, StructDefinition, FieldDefinition,
                                   Endianness, TrackRecordParser)
from parsers.text_parser import TextParser
from parsers.file_detector import FileDetector


class TestBinaryParser:
    """Tests for binary parser."""
    
    def test_struct_definition(self):
        """Test struct definition creation."""
        fields = [
            FieldDefinition('timestamp', 'uint64_t'),
            FieldDefinition('value', 'float'),
        ]
        struct_def = StructDefinition('TestStruct', fields, packed=True)
        
        assert struct_def.get_format_string() == '<Qf'
        assert struct_def.get_size() == 12
    
    def test_parse_simple_struct(self):
        """Test parsing simple struct."""
        # Create test data
        fields = [
            FieldDefinition('id', 'uint32_t'),
            FieldDefinition('value', 'float'),
        ]
        struct_def = StructDefinition('TestStruct', fields, packed=True)
        parser = BinaryParser(struct_def)
        
        # Write test file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            f.write(struct.pack('<If', 1, 3.14))
            f.write(struct.pack('<If', 2, 2.71))
            temp_path = f.name
        
        try:
            # Parse
            records = parser.parse_file(temp_path)
            
            assert len(records) == 2
            assert records[0]['id'] == 1
            assert records[0]['value'] == pytest.approx(3.14, rel=1e-5)
            assert records[1]['id'] == 2
            assert records[1]['value'] == pytest.approx(2.71, rel=1e-5)
        finally:
            Path(temp_path).unlink()
    
    def test_track_record_parser(self):
        """Test default track record parser."""
        parser = TrackRecordParser()
        
        # Create test file with one record
        struct_def = parser.struct_def
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            # Pack a track record
            record = struct.pack(
                '<QIfffffff3f3f3f3f3fB',
                1000000,  # timestamp
                1,        # track_id
                10000.0,  # range
                0.5,      # azimuth
                0.2,      # elevation
                -50.0,    # range_rate
                30.0,     # snr
                10.0,     # rcs
                1000.0,   # doppler
                100.0, 200.0, 5000.0,  # position
                50.0, 30.0, 10.0,      # velocity
                0.0, 0.0, 0.0,         # acceleration
                1.0, 1.0, 1.0,         # pos_error
                0.5, 0.5, 0.5,         # vel_error
                1                      # measurement_valid
            )
            f.write(record)
            temp_path = f.name
        
        try:
            df = parser.parse_to_dataframe(temp_path)
            
            assert len(df) == 1
            assert df['track_id'].iloc[0] == 1
            assert df['range'].iloc[0] == pytest.approx(10000.0)
        finally:
            Path(temp_path).unlink()


class TestTextParser:
    """Tests for text parser."""
    
    def test_csv_parsing(self):
        """Test CSV parsing."""
        parser = TextParser()
        
        # Create test CSV
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("track_id,timestamp,range,azimuth\n")
            f.write("1,1000,5000.0,0.5\n")
            f.write("1,2000,4950.0,0.52\n")
            temp_path = f.name
        
        try:
            df = parser.parse_file(temp_path)
            
            assert len(df) == 2
            assert 'track_id' in df.columns
            assert df['track_id'].iloc[0] == 1
        finally:
            Path(temp_path).unlink()
    
    def test_format_detection(self):
        """Test format auto-detection."""
        parser = TextParser()
        
        # CSV
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("a,b,c\n1,2,3\n")
            temp_path = f.name
        
        try:
            format_detected = parser.detect_format(temp_path)
            assert format_detected == 'csv'
        finally:
            Path(temp_path).unlink()


class TestFileDetector:
    """Tests for file detector."""
    
    def test_binary_detection(self):
        """Test binary file detection."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            f.write(b'\x00\x01\x02\x03\x04\x05')
            temp_path = f.name
        
        try:
            file_type, _ = FileDetector.detect_file_type(temp_path)
            assert file_type == 'binary'
        finally:
            Path(temp_path).unlink()
    
    def test_text_detection(self):
        """Test text file detection."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("This is a text file\n")
            temp_path = f.name
        
        try:
            file_type, _ = FileDetector.detect_file_type(temp_path)
            assert file_type == 'text'
        finally:
            Path(temp_path).unlink()
