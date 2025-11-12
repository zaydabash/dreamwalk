"""
Pytest configuration and shared fixtures
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.setex = AsyncMock(return_value=True)
    redis_mock.keys = AsyncMock(return_value=[])
    redis_mock.lrange = AsyncMock(return_value=[])
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.info = AsyncMock(return_value={"used_memory_human": "1MB", "connected_clients": 0})
    redis_mock.close = AsyncMock()
    return redis_mock


@pytest.fixture
def mock_http_client():
    """Mock HTTP client"""
    client_mock = AsyncMock()
    client_mock.get = AsyncMock()
    client_mock.post = AsyncMock()
    client_mock.aclose = AsyncMock()
    return client_mock


@pytest.fixture
def mock_eeg_data():
    """Mock EEG signal data"""
    import numpy as np
    return {
        "channels": ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"],
        "sample_rate": 250,
        "data": np.random.randn(8, 2500),  # 8 channels, 10 seconds at 250 Hz
        "timestamp": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def mock_neural_features():
    """Mock neural feature extraction results"""
    import numpy as np
    return {
        "band_powers": {
            "alpha": np.random.rand(8),
            "beta": np.random.rand(8),
            "gamma": np.random.rand(8),
            "theta": np.random.rand(8),
            "delta": np.random.rand(8)
        },
        "hjorth_params": {
            "activity": np.random.rand(8),
            "mobility": np.random.rand(8),
            "complexity": np.random.rand(8)
        },
        "asymmetry": np.random.rand(4),
        "connectivity": np.random.rand(8, 8),
        "timestamp": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def mock_world_state():
    """Mock world state data"""
    return {
        "session_id": "test-session-123",
        "timestamp": "2024-01-01T00:00:00Z",
        "world_state": {
            "biome_type": "forest",
            "weather_intensity": 0.5,
            "lighting_mood": "calm",
            "color_palette": [0.3, 0.6, 0.4],
            "object_density": 0.5,
            "structure_level": 0.7,
            "ambient_volume": 0.6,
            "music_intensity": 0.4,
            "sound_effects": [],
            "change_rate": 0.1,
            "morph_speed": 1.0
        },
        "emotional_state": {
            "valence": 0.6,
            "arousal": 0.4,
            "dominance": 0.5
        },
        "motifs": ["peaceful", "natural"]
    }

