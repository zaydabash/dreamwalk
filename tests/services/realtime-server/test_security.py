"""
Unit tests for the realtime-server security utilities: session ID validation,
JSON input validation, string sanitization, rate limiting, and client IP
resolution.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

# Add service to path (hyphenated directory can't be imported as a package)
service_path = Path(__file__).parent.parent.parent.parent / "services" / "realtime-server"
sys.path.insert(0, str(service_path))

from utils.security import (  # noqa: E402
    check_rate_limit,
    get_client_ip,
    rate_limit_storage,
    sanitize_string,
    validate_json_input,
    validate_session_id,
)


@pytest.mark.unit
class TestValidateSessionId:
    def test_accepts_valid_session_id(self):
        assert validate_session_id("session-123_abc") == "session-123_abc"

    def test_rejects_empty_session_id(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_session_id("")
        assert exc_info.value.status_code == 400

    def test_rejects_invalid_characters(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_session_id("session/../etc")
        assert exc_info.value.status_code == 400

    def test_rejects_overly_long_session_id(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_session_id("a" * 101)
        assert exc_info.value.status_code == 400


@pytest.mark.unit
class TestValidateJsonInput:
    def test_accepts_valid_payload_with_required_fields(self):
        data = {"session_id": "abc", "signal_type": "eeg"}
        assert validate_json_input(data, required_fields=["session_id"]) == data

    def test_rejects_non_dict_input(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_json_input("not-a-dict", required_fields=[])
        assert exc_info.value.status_code == 400

    def test_rejects_missing_required_field(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_json_input({"foo": "bar"}, required_fields=["session_id"])
        assert exc_info.value.status_code == 400

    def test_rejects_oversized_payload(self):
        data = {"value": "x" * 100}
        with pytest.raises(HTTPException) as exc_info:
            validate_json_input(data, required_fields=[], max_size=10)
        assert exc_info.value.status_code == 400


@pytest.mark.unit
class TestSanitizeString:
    def test_strips_null_bytes_and_control_characters(self):
        assert sanitize_string("hello\x00world\x01") == "helloworld"

    def test_preserves_newlines_and_tabs(self):
        assert sanitize_string("line1\nline2\ttabbed") == "line1\nline2\ttabbed"

    def test_rejects_non_string_input(self):
        with pytest.raises(HTTPException) as exc_info:
            sanitize_string(123)
        assert exc_info.value.status_code == 400

    def test_rejects_overly_long_string(self):
        with pytest.raises(HTTPException) as exc_info:
            sanitize_string("a" * 10, max_length=5)
        assert exc_info.value.status_code == 400


@pytest.mark.unit
class TestCheckRateLimit:
    def setup_method(self):
        rate_limit_storage.clear()

    def test_allows_requests_under_limit(self):
        for _ in range(5):
            assert check_rate_limit("client-1", max_requests=5, window_seconds=60) is True

    def test_blocks_requests_over_limit(self):
        for _ in range(5):
            check_rate_limit("client-2", max_requests=5, window_seconds=60)

        assert check_rate_limit("client-2", max_requests=5, window_seconds=60) is False

    def test_tracks_identifiers_independently(self):
        for _ in range(5):
            check_rate_limit("client-3", max_requests=5, window_seconds=60)

        assert check_rate_limit("client-4", max_requests=5, window_seconds=60) is True


@pytest.mark.unit
class TestGetClientIp:
    def _make_request(self, headers=None, client_host="127.0.0.1"):
        request = MagicMock()
        request.headers = headers or {}
        request.client = MagicMock()
        request.client.host = client_host
        return request

    def test_prefers_x_forwarded_for(self):
        request = self._make_request(headers={"X-Forwarded-For": "1.2.3.4, 5.6.7.8"})
        assert get_client_ip(request) == "1.2.3.4"

    def test_falls_back_to_x_real_ip(self):
        request = self._make_request(headers={"X-Real-IP": "9.9.9.9"})
        assert get_client_ip(request) == "9.9.9.9"

    def test_falls_back_to_direct_client_host(self):
        request = self._make_request()
        assert get_client_ip(request) == "127.0.0.1"

    def test_returns_unknown_when_no_client_info(self):
        request = self._make_request()
        request.client = None
        assert get_client_ip(request) == "unknown"
