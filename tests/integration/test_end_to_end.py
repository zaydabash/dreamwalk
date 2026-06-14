"""
Integration tests covering the signal-processing -> neural-decoding ->
world-state/narrative pipeline across services.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Hyphenated service directories can't be imported as dotted packages, so add
# each service directory to the path. The `models` packages have no
# __init__.py, so they merge as namespace packages across services.
for service in ("signal-processor", "neural-decoder", "realtime-server"):
    sys.path.insert(0, str(PROJECT_ROOT / "services" / service))

from streamers.mock_streamer import MockStreamer  # noqa: E402
from processors.eeg_processor import EEGProcessor  # noqa: E402
from feature_extractors.neural_features import NeuralFeatureExtractor  # noqa: E402
from models.signal_models import EEGConfig, ProcessingConfig  # noqa: E402

from decoders.eeg_to_clip import EEGToCLIPDecoder  # noqa: E402
from decoders.emotion_classifier import EmotionClassifier  # noqa: E402
from decoders.motif_detector import MotifDetector  # noqa: E402
from models.decoder_models import DecoderConfig  # noqa: E402

import main as realtime_main  # noqa: E402


def _mock_response(status_code=200, json_data=None, text=""):
    response = MagicMock()
    response.status_code = status_code
    response.json = MagicMock(return_value=json_data or {})
    response.text = text
    return response


@pytest.mark.integration
class TestEndToEnd:
    """Test end-to-end workflows across services"""

    @pytest.mark.asyncio
    async def test_signal_processing_pipeline(self):
        """Mock EEG stream -> preprocessing -> feature extraction"""
        streamer = MockStreamer({"sampling_rate": 250, "window_size": 4.0})
        raw_signal = await streamer._generate_signal_window()

        eeg_config = EEGConfig(sampling_rate=raw_signal.sampling_rate, channels=raw_signal.channels)
        processor = EEGProcessor(eeg_config)
        processed_signal = await processor.process(raw_signal)

        assert processed_signal.data.shape == raw_signal.data.shape

        extractor = NeuralFeatureExtractor(ProcessingConfig())
        features = await extractor.extract(processed_signal)

        n_channels = len(raw_signal.channels)
        assert len(features.alpha_power) == n_channels
        assert len(features.hjorth_activity) == n_channels

    @pytest.mark.asyncio
    async def test_neural_decoding_workflow(self):
        """Processed neural features -> emotion, CLIP embedding, and motifs"""
        streamer = MockStreamer({"sampling_rate": 250, "window_size": 4.0})
        raw_signal = await streamer._generate_signal_window()

        eeg_config = EEGConfig(sampling_rate=raw_signal.sampling_rate, channels=raw_signal.channels)
        processor = EEGProcessor(eeg_config)
        processed_signal = await processor.process(raw_signal)

        extractor = NeuralFeatureExtractor(ProcessingConfig())
        features = await extractor.extract(processed_signal)
        feature_dict = features.dict()

        decoder_config = DecoderConfig(device="cpu")

        emotion_classifier = EmotionClassifier(config=decoder_config)
        emotional_state = await emotion_classifier.classify(feature_dict)
        assert -1.0 <= emotional_state.valence <= 1.0
        assert 0.0 <= emotional_state.arousal <= 1.0

        clip_decoder = EEGToCLIPDecoder(config=decoder_config)
        embedding = await clip_decoder.decode(feature_dict)
        assert embedding.shape == (clip_decoder.output_dim,)

        motif_detector = MotifDetector(config=decoder_config)
        motifs = await motif_detector.detect(feature_dict)
        assert isinstance(motifs, list)

    @pytest.mark.asyncio
    async def test_world_generation_workflow(self, monkeypatch):
        """World state (with emotional state + motifs) -> ambient narrative"""
        world_state = {
            "session_id": "session-int-1",
            "biome_type": "forest",
            "emotional_state": {"valence": 0.4, "arousal": 0.3, "dominance": 0.5},
            "motifs": [{"motif_type": "relaxation"}],
            "semantic_embedding": [0.1] * 8,
        }

        async def fake_post(url, json=None, timeout=None):
            assert url.endswith("/generate")
            neural_state = json["neural_state"]
            assert neural_state["valence"] == 0.4
            assert neural_state["motif_tags"] == ["relaxation"]
            return _mock_response(status_code=200, json_data={"ambient_text": "a calm forest clearing"})

        client = AsyncMock()
        client.post = AsyncMock(side_effect=fake_post)
        monkeypatch.setattr(realtime_main, "http_client", client)

        narrative = await realtime_main._generate_narrative("session-int-1", world_state)

        assert narrative == "a calm forest clearing"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_full_system_integration(self, monkeypatch):
        """Full pipeline: mock EEG -> features -> decoding -> narrative"""
        streamer = MockStreamer({"sampling_rate": 250, "window_size": 4.0})
        raw_signal = await streamer._generate_signal_window()

        eeg_config = EEGConfig(sampling_rate=raw_signal.sampling_rate, channels=raw_signal.channels)
        processor = EEGProcessor(eeg_config)
        processed_signal = await processor.process(raw_signal)

        extractor = NeuralFeatureExtractor(ProcessingConfig())
        features = await extractor.extract(processed_signal)
        feature_dict = features.dict()

        decoder_config = DecoderConfig(device="cpu")
        emotion_classifier = EmotionClassifier(config=decoder_config)
        emotional_state = await emotion_classifier.classify(feature_dict)

        motif_detector = MotifDetector(config=decoder_config)
        motifs = await motif_detector.detect(feature_dict)

        clip_decoder = EEGToCLIPDecoder(config=decoder_config)
        embedding = await clip_decoder.decode(feature_dict)

        world_state = {
            "session_id": "session-full-1",
            "biome_type": "neutral",
            "emotional_state": emotional_state.dict(),
            "motifs": [motif.dict() for motif in motifs],
            "semantic_embedding": embedding.tolist(),
        }

        async def fake_post(url, json=None, timeout=None):
            return _mock_response(status_code=200, json_data={"ambient_text": "drifting through the unknown"})

        client = AsyncMock()
        client.post = AsyncMock(side_effect=fake_post)
        monkeypatch.setattr(realtime_main, "http_client", client)

        narrative = await realtime_main._generate_narrative("session-full-1", world_state)

        assert narrative == "drifting through the unknown"
