"""
Unit tests for binary and text parsers.
"""
import pytest
import tempfile
import struct
import pandas as pd
from pathlib import Path

from parsers.binary.struct_parser import (
    StructParser, TrackRecordParser, FieldDef, Endianness
)
from parsers.text.text_parser import TextParser, TextFormat
from parsers.file_router import FileRouter


class TestBinaryParser:
    """Test binary parser"""
    
    def test_simple_struct_parsing(self):
        """Test parsing simple struct"""
        struct_def = [
            FieldDef("id", "uint32_t"),
            FieldDef("value", "float"),
        ]
        
        parser = StructParser(struct_def, Endianness.LITTLE, packed=True)
        
        # Create test data
        test_data = struct.pack('<If', 42, 3.14)
        
        record = parser.parse_record(test_data)
        
        assert record['id'] == 42
        assert abs(record['value'] - 3.14) < 0.01
    
    def test_array_fields(self):
        """Test parsing arrays"""
        struct_def = [
            FieldDef("id", "uint32_t"),
            FieldDef("values", "float", 3),
        ]
        
        parser = StructParser(struct_def, Endianness.LITTLE, packed=True)
        
        test_data = struct.pack('<Ifff', 1, 1.0, 2.0, 3.0)
        
        record = parser.parse_record(test_data)
        
        assert record['id'] == 1
        assert len(record['values']) == 3
        assert record['values'] == [1.0, 2.0, 3.0]
    
    def test_endianness(self):
        """Test different endianness"""
        struct_def = [FieldDef("value", "uint32_t")]
        
        # Little endian
        parser_le = StructParser(struct_def, Endianness.LITTLE)
        data_le = struct.pack('<I', 0x12345678)
        assert parser_le.parse_record(data_le)['value'] == 0x12345678
        
        # Big endian
        parser_be = StructParser(struct_def, Endianness.BIG)
        data_be = struct.pack('>I', 0x12345678)
        assert parser_be.parse_record(data_be)['value'] == 0x12345678
    
    def test_track_record_parser(self):
        """Test track record parser"""
        parser = TrackRecordParser(
            TrackRecordParser.get_standard_track_struct(),
            Endianness.LITTLE,
            packed=True
        )
        
        # Create sample track record
        track_id = 1
        timestamp = 1000
        range_val = 5000.0
        azimuth = 0.5
        elevation = 0.2
        
        # Pack data
        data = struct.pack(
            '<IQffffffffffffffffI',
            track_id, timestamp, range_val, azimuth, elevation,
            -100.0, 20.0, 10.0, 0.5,  # range_rate, snr, rcs, doppler
            1000.0, 2000.0, 3000.0,    # position
            -50.0, -30.0, 10.0,        # velocity
            0.0, 0.0, 0.0,             # acceleration
            5.0, 5.0, 2.0,             # pos_error
            1.0, 1.0, 0.5,             # vel_error
            0xFF                        # valid_flags
        )
        
        record = parser.parse_record(data)
        
        assert record['track_id'] == track_id
        assert record['timestamp'] == timestamp
        assert abs(record['range'] - range_val) < 0.1


class TestTextParser:
    """Test text parser"""
    
    def test_csv_detection(self):
        """Test CSV format detection"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("track_id,timestamp,range\n")
            f.write("1,100,5000\n")
            filepath = f.name
        
        try:
            format = TextParser.detect_format(filepath)
            assert format == TextFormat.CSV
        finally:
            Path(filepath).unlink()
    
    def test_csv_parsing(self):
        """Test CSV parsing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("track_id,timestamp,range\n")
            f.write("1,100,5000\n")
            f.write("1,101,5100\n")
            filepath = f.name
        
        try:
            df = TextParser.parse_csv(filepath)
            assert len(df) == 2
            assert 'track_id' in df.columns
            assert df['track_id'].iloc[0] == 1
        finally:
            Path(filepath).unlink()
    
    def test_whitespace_parsing(self):
        """Test whitespace-delimited parsing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("track_id timestamp range\n")
            f.write("1 100 5000\n")
            f.write("2 200 6000\n")
            filepath = f.name
        
        try:
            df = TextParser.parse_whitespace(filepath)
            assert len(df) == 2
            assert 'range' in df.columns
        finally:
            Path(filepath).unlink()
    
    def test_column_normalization(self):
        """Test column name normalization"""
        df = pd.DataFrame({
            'TrackID': [1, 2],
            'Time': [100, 200],
            'R': [5000, 6000]
        })
        
        normalized = TextParser.normalize_columns(df)
        
        assert 'track_id' in normalized.columns
        assert 'timestamp' in normalized.columns
        assert 'range' in normalized.columns


class TestFileRouter:
    """Test file router"""
    
    def test_file_type_detection(self):
        """Test file type detection"""
        # Text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("test,data\n")
            text_file = f.name
        
        try:
            assert FileRouter.detect_file_type(text_file) == 'text'
        finally:
            Path(text_file).unlink()
        
        # Binary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False) as f:
            f.write(b'\x00\x01\x02\x03\x04\x05')
            bin_file = f.name
        
        try:
            assert FileRouter.detect_file_type(bin_file) == 'binary'
        finally:
            Path(bin_file).unlink()
    
    def test_data_validation(self):
        """Test track data validation"""
        # Valid data
        valid_df = pd.DataFrame({
            'track_id': [1, 1],
            'timestamp': [100, 101]
        })
        assert FileRouter.validate_track_data(valid_df)
        
        # Invalid data (missing track_id)
        invalid_df = pd.DataFrame({
            'timestamp': [100, 101]
        })
        assert not FileRouter.validate_track_data(invalid_df)
