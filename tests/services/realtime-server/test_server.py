"""
Unit tests for the real-time server orchestration logic: service health
checks, narrative generation, and neural-feature-to-world-state decoding.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add service to path (hyphenated directory can't be imported as a package)
service_path = Path(__file__).parent.parent.parent.parent / "services" / "realtime-server"
sys.path.insert(0, str(service_path))

import main as realtime_main  # noqa: E402


def _mock_response(status_code=200, json_data=None, text=""):
    response = MagicMock()
    response.status_code = status_code
    response.json = MagicMock(return_value=json_data or {})
    response.text = text
    return response


@pytest.mark.unit
class TestServiceHealth:
    """Tests for _check_service_health and the /health endpoint logic"""

    @pytest.mark.asyncio
    async def test_check_service_health_returns_true_on_200(self, monkeypatch):
        client = AsyncMock()
        client.get = AsyncMock(return_value=_mock_response(status_code=200))
        monkeypatch.setattr(realtime_main, "http_client", client)

        assert await realtime_main._check_service_health("http://service") is True

    @pytest.mark.asyncio
    async def test_check_service_health_returns_false_on_error_status(self, monkeypatch):
        client = AsyncMock()
        client.get = AsyncMock(return_value=_mock_response(status_code=500))
        monkeypatch.setattr(realtime_main, "http_client", client)

        assert await realtime_main._check_service_health("http://service") is False

    @pytest.mark.asyncio
    async def test_check_service_health_returns_false_on_exception(self, monkeypatch):
        client = AsyncMock()
        client.get = AsyncMock(side_effect=ConnectionError("boom"))
        monkeypatch.setattr(realtime_main, "http_client", client)

        assert await realtime_main._check_service_health("http://service") is False

    @pytest.mark.asyncio
    async def test_health_check_reports_service_statuses(self, monkeypatch):
        client = AsyncMock()
        client.get = AsyncMock(return_value=_mock_response(status_code=200))
        monkeypatch.setattr(realtime_main, "http_client", client)

        result = await realtime_main.health_check()

        assert result["status"] == "healthy"
        assert result["services"]["signal_processor"] is True
        assert result["services"]["neural_decoder"] is True
        assert result["services"]["texture_generator"] is True
        assert result["services"]["narrative_layer"] is True


@pytest.mark.unit
class TestGenerateNarrative:
    """Tests for the narrative-layer integration contract"""

    @pytest.mark.asyncio
    async def test_sends_neural_state_shaped_request_and_parses_ambient_text(self, monkeypatch):
        captured = {}

        async def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            return _mock_response(status_code=200, json_data={"ambient_text": "a quiet glade"})

        client = AsyncMock()
        client.post = AsyncMock(side_effect=fake_post)
        monkeypatch.setattr(realtime_main, "http_client", client)

        world_state = {
            "emotional_state": {"valence": 0.6, "arousal": 0.4, "dominance": 0.5},
            "motifs": [{"motif_type": "focus"}, {"motif_type": "flow_state"}],
            "semantic_embedding": [0.1, 0.2, 0.3],
        }

        narrative = await realtime_main._generate_narrative("session-1", world_state)

        assert narrative == "a quiet glade"
        assert captured["url"] == f"{realtime_main.NARRATIVE_LAYER_URL}/generate"

        sent_neural_state = captured["json"]["neural_state"]
        assert sent_neural_state["valence"] == 0.6
        assert sent_neural_state["arousal"] == 0.4
        assert sent_neural_state["dominance"] == 0.5
        assert sent_neural_state["motif_tags"] == ["focus", "flow_state"]
        assert sent_neural_state["latent_vector"] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_returns_empty_string_on_non_200(self, monkeypatch):
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response(status_code=422))
        monkeypatch.setattr(realtime_main, "http_client", client)

        narrative = await realtime_main._generate_narrative("session-1", {"emotional_state": {}})

        assert narrative == ""

    @pytest.mark.asyncio
    async def test_returns_empty_string_on_exception(self, monkeypatch):
        client = AsyncMock()
        client.post = AsyncMock(side_effect=ConnectionError("boom"))
        monkeypatch.setattr(realtime_main, "http_client", client)

        narrative = await realtime_main._generate_narrative("session-1", {"emotional_state": {}})

        assert narrative == ""

    @pytest.mark.asyncio
    async def test_handles_missing_optional_world_state_fields(self, monkeypatch):
        captured = {}

        async def fake_post(url, json=None, timeout=None):
            captured["json"] = json
            return _mock_response(status_code=200, json_data={"ambient_text": "stillness"})

        client = AsyncMock()
        client.post = AsyncMock(side_effect=fake_post)
        monkeypatch.setattr(realtime_main, "http_client", client)

        narrative = await realtime_main._generate_narrative("session-1", {"emotional_state": {}})

        assert narrative == "stillness"
        sent_neural_state = captured["json"]["neural_state"]
        assert sent_neural_state["valence"] == 0.0
        assert sent_neural_state["motif_tags"] == []
        assert sent_neural_state["latent_vector"] == []


@pytest.mark.unit
class TestDecodeToWorldState:
    """Tests for the neural-decoder integration and resulting world state"""

    @pytest.mark.asyncio
    async def test_returns_world_state_update_from_decoder_response(self, monkeypatch):
        decoder_response = {
            "world_state": {
                "session_id": "session-1",
                "biome_type": "forest",
                "emotional_state": {"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
                "motifs": [],
                "semantic_embedding": [],
            }
        }

        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response(status_code=200, json_data=decoder_response))
        monkeypatch.setattr(realtime_main, "http_client", client)
        monkeypatch.setattr(realtime_main.config, "enable_texture_generation", False)
        monkeypatch.setattr(realtime_main.config, "enable_narrative", False)

        result = await realtime_main._decode_to_world_state("session-1", {"alpha_power": [0.1]})

        assert result.session_id == "session-1"
        assert result.world_state["biome_type"] == "forest"

    @pytest.mark.asyncio
    async def test_returns_default_world_state_on_decoder_failure(self, monkeypatch):
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response(status_code=500, text="decoder error"))
        monkeypatch.setattr(realtime_main, "http_client", client)

        result = await realtime_main._decode_to_world_state("session-1", {})

        assert result.session_id == "session-1"
        assert result.world_state["biome_type"] == "neutral"

    @pytest.mark.asyncio
    async def test_attaches_narrative_when_enabled(self, monkeypatch):
        decoder_response = {
            "world_state": {
                "session_id": "session-1",
                "biome_type": "forest",
                "emotional_state": {"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
                "motifs": [],
                "semantic_embedding": [],
            }
        }

        async def fake_post(url, json=None, timeout=None):
            if url.endswith("/decode"):
                return _mock_response(status_code=200, json_data=decoder_response)
            if url.endswith("/generate"):
                return _mock_response(status_code=200, json_data={"ambient_text": "a quiet glade"})
            raise AssertionError(f"unexpected url: {url}")

        client = AsyncMock()
        client.post = AsyncMock(side_effect=fake_post)
        monkeypatch.setattr(realtime_main, "http_client", client)
        monkeypatch.setattr(realtime_main.config, "enable_texture_generation", False)
        monkeypatch.setattr(realtime_main.config, "enable_narrative", True)

        result = await realtime_main._decode_to_world_state("session-1", {})

        assert result.world_state["narrative"] == "a quiet glade"
