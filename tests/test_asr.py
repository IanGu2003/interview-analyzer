"""Tests for ASR utility module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "streamlit_app"))

from utils.asr import (
    check_ffmpeg,
    convert_to_standard_wav,
    get_file_size_mb,
    check_aliyun_asr_deps,
)


class TestASRUtilities:
    """Test ASR utility functions"""

    def test_get_file_size_mb(self):
        """Test file size calculation"""
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(b"\x00" * 1024 * 1024)  # 1MB
            tmp_path = f.name
        try:
            size = get_file_size_mb(tmp_path)
            assert abs(size - 1.0) < 0.1  # roughly 1MB
        finally:
            os.unlink(tmp_path)

    def test_get_file_size_mb_empty(self):
        """Test file size for empty file"""
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            tmp_path = f.name
        try:
            size = get_file_size_mb(tmp_path)
            assert size >= 0
        finally:
            os.unlink(tmp_path)

    def test_check_aliyun_deps(self):
        """Test Aliyun dependency check returns a tuple"""
        ok, msg = check_aliyun_asr_deps()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_convert_to_standard_wav_no_input(self):
        """Test conversion with non-existent file returns None"""
        result = convert_to_standard_wav("/nonexistent/audio.mp3")
        assert result is None