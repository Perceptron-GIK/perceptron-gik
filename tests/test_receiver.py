"""Tests for the receiver module's helper functions.

These tests cover:
- get_session_file: session ID management and file path generation
- _prepare_csv_file: CSV pre-initialisation (Issue #3 fix)
- print_data: data unpacking and console display
- handler_closure: BLE packet handling and queue insertion
"""

import asyncio
import os
import struct
import tempfile
import pytest

# We need to mock the keyboard import before importing receiver
import sys
from unittest.mock import MagicMock, patch
import threading

# Create a mock for the keyboard_ext module so we can import receiver
# without needing the 'keyboard' package
mock_keyboard_ext = MagicMock()
mock_keyboard_ext.stop_event = threading.Event()
sys.modules['src.keyboard.keyboard_ext'] = mock_keyboard_ext
sys.modules['src.keyboard'] = MagicMock()

# Mock bleak so receiver.py can be imported without BLE hardware
sys.modules['bleak'] = MagicMock()

# Now we can safely import from receiver
# We need to import receiver as a module rather than running it
import importlib
import types


def _load_receiver_module():
    """Load receiver.py as a module without executing asyncio.run(main())."""
    receiver_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "receiver.py",
    )
    with open(receiver_path) as f:
        source = f.read()
    # Remove the final asyncio.run(main()) call so we can import cleanly
    source = source.replace("asyncio.run(main())", "# asyncio.run(main())")
    # Also patch the import line
    source = source.replace(
        "from src.keyboard.keyboard_ext import start_keyboard, stop_event",
        "from src.keyboard.keyboard_ext import start_keyboard, stop_event",
    )
    mod = types.ModuleType("receiver")
    mod.__file__ = receiver_path
    exec(compile(source, receiver_path, "exec"), mod.__dict__)
    return mod


receiver = _load_receiver_module()


class TestGetSessionFile:
    """Tests for get_session_file()."""

    def test_creates_first_session(self, tmp_path, monkeypatch):
        """First call should create session 1."""
        monkeypatch.setattr(receiver, "DATA_DIR", str(tmp_path))
        path = receiver.get_session_file("Left")
        assert path == os.path.join(str(tmp_path), "Left_1.csv")
        # Metadata file should exist with '1'
        meta = tmp_path / "metadata_Left.txt"
        assert meta.read_text().strip() == "1"

    def test_increments_session_id(self, tmp_path, monkeypatch):
        """Subsequent calls should increment session ID."""
        monkeypatch.setattr(receiver, "DATA_DIR", str(tmp_path))
        # Create initial metadata
        meta = tmp_path / "metadata_Right.txt"
        meta.write_text("3\n")

        path = receiver.get_session_file("Right")
        assert path == os.path.join(str(tmp_path), "Right_4.csv")
        assert meta.read_text().strip() == "4"

    def test_override_session_id(self, tmp_path, monkeypatch):
        """When OVERRIDE_SESSION_ID is True, use configured IDs."""
        monkeypatch.setattr(receiver, "DATA_DIR", str(tmp_path))
        monkeypatch.setattr(receiver, "OVERRIDE_SESSION_ID", True)
        monkeypatch.setattr(receiver, "LEFT_SESSION_ID", 42)

        path = receiver.get_session_file("Left")
        assert path == os.path.join(str(tmp_path), "Left_42.csv")


class TestPrepareCsvFile:
    """Tests for _prepare_csv_file() – Issue #3 fix."""

    def test_creates_file_with_header(self, tmp_path, monkeypatch):
        """CSV file should be created with the data header before BLE connection."""
        monkeypatch.setattr(receiver, "DATA_DIR", str(tmp_path))
        path = receiver._prepare_csv_file("Left")
        assert os.path.exists(path)
        with open(path) as f:
            header = f.readline().strip()
        assert header == receiver.DATA_HEADER

    def test_does_not_overwrite_existing(self, tmp_path, monkeypatch):
        """If the file already exists, it should not be overwritten."""
        monkeypatch.setattr(receiver, "DATA_DIR", str(tmp_path))
        # Create a dummy file
        meta = tmp_path / "metadata_Left.txt"
        meta.write_text("1\n")
        csv_file = tmp_path / "Left_2.csv"
        csv_file.write_text("existing data\n")

        path = receiver._prepare_csv_file("Left")
        with open(path) as f:
            content = f.read()
        assert "existing data" in content


class TestPrintData:
    """Tests for print_data()."""

    def _make_sample_data(self):
        """Create a valid 42-element list matching PACKER_DTYPE_DEF."""
        data = [1]  # sample_id (uint32 stored as int)
        data.extend([0.0] * 6)  # base IMU
        for _ in range(5):  # 5 fingers
            data.extend([0.0] * 6)  # accel + gyro
            data.append(0)  # FSR bool
        return data

    def test_print_data_runs_without_error(self, capsys):
        """print_data should not raise with valid input."""
        data = self._make_sample_data()
        receiver.print_data(data)
        captured = capsys.readouterr()
        assert "id=1" in captured.out


class TestHandlerClosure:
    """Tests for handler_closure()."""

    def test_valid_packet_queued(self):
        """Valid 153-byte packet should be unpacked and put on the queue."""
        queue = asyncio.Queue(10)
        handler = receiver.handler_closure(queue, "Left")

        # Build a valid 153-byte packet
        values = [1]  # sample_id uint32
        values.extend([0.0] * 6)  # base IMU floats
        for _ in range(5):
            values.extend([0.0] * 6)  # finger floats
            values.append(0)  # FSR uint8
        packet = struct.pack(receiver.PACKER_DTYPE_DEF, *values)
        assert len(packet) == 153

        handler(None, packet)
        assert not queue.empty()
        data, t = queue.get_nowait()
        assert data[0] == 1  # sample_id

    def test_invalid_length_ignored(self):
        """Packets not exactly 153 bytes should be dropped."""
        queue = asyncio.Queue(10)
        handler = receiver.handler_closure(queue, "Right")
        handler(None, b"\x00" * 100)
        assert queue.empty()


class TestCsvWriter:
    """Tests for the async CSV writer."""

    def test_csv_writer_writes_data(self, tmp_path):
        """_csv_writer should consume from the queue and write CSV rows."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(receiver.DATA_HEADER + "\n")

        async def _run():
            queue = asyncio.Queue(10)
            # Start writer
            task = asyncio.create_task(
                receiver._csv_writer(queue, str(csv_file))
            )
            # Put data
            sample = list(range(42))
            await queue.put((sample, 1234567890.0))
            # Give it time to write
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())

        with open(csv_file) as f:
            lines = f.readlines()
        assert len(lines) >= 2  # header + at least 1 data row
