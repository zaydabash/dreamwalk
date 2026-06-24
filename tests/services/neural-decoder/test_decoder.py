"""
Unit tests for neural decoder components: EEG-to-CLIP embedding decoder,
emotion classifier, motif detector, and synthetic data generator.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add service to path (hyphenated directory can't be imported as a package)
service_path = Path(__file__).parent.parent.parent.parent / "services" / "neural-decoder"
sys.path.insert(0, str(service_path))

from decoders.eeg_to_clip import EEGToCLIPDecoder  # noqa: E402
from decoders.emotion_classifier import EmotionClassifier  # noqa: E402
from decoders.motif_detector import MotifDetector, SUPPORTED_MOTIFS  # noqa: E402
from models.decoder_models import DecoderConfig  # noqa: E402
from synthetic_data.generator import SyntheticDataGenerator  # noqa: E402


def _sample_features(n_channels: int = 8) -> dict:
    """A complete, well-formed neural feature dictionary"""
    return {
        "delta_power": [0.1] * n_channels,
        "theta_power": [0.2] * n_channels,
        "alpha_power": [0.3] * n_channels,
        "beta_power": [0.15] * n_channels,
        "gamma_power": [0.05] * n_channels,
        "hjorth_activity": [1.0] * n_channels,
        "hjorth_mobility": [0.5] * n_channels,
        "hjorth_complexity": [1.5] * n_channels,
        "frontal_asymmetry": 0.1,
        "parietal_asymmetry": -0.05,
        "artifact_ratio": 0.02,
        "eye_blink_count": 1,
        "mean_amplitude": [0.0] * n_channels,
        "std_amplitude": [1.0] * n_channels,
        "skewness": [0.0] * n_channels,
        "kurtosis": [3.0] * n_channels,
    }


@pytest.fixture
def decoder_config():
    return DecoderConfig(device="cpu")


@pytest.mark.unit
class TestEEGToCLIPDecoder:
    """Tests for the EEG-to-CLIP embedding decoder"""

    @pytest.mark.asyncio
    async def test_decode_untrained_returns_clip_sized_embedding(self, decoder_config):
        decoder = EEGToCLIPDecoder(config=decoder_config)
        embedding = await decoder.decode(_sample_features())

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (decoder.output_dim,)

    def test_extract_feature_vector_has_expected_length(self, decoder_config):
        decoder = EEGToCLIPDecoder(config=decoder_config)
        vector = decoder._extract_feature_vector(_sample_features())

        assert vector.shape[0] == decoder.input_dim

    def test_extract_feature_vector_fills_missing_bands_with_zeros(self, decoder_config):
        decoder = EEGToCLIPDecoder(config=decoder_config)
        vector = decoder._extract_feature_vector({})

        assert vector.shape[0] == decoder.input_dim
        assert np.all(vector == 0.0)

    def test_get_model_info_reports_untrained_state(self, decoder_config):
        decoder = EEGToCLIPDecoder(config=decoder_config)
        info = decoder.get_model_info()

        assert info["is_trained"] is False
        assert info["output_dim"] == 512


@pytest.mark.unit
class TestEmotionClassifier:
    """Tests for the emotion classifier"""

    @pytest.mark.asyncio
    async def test_classify_untrained_returns_neutral_state(self, decoder_config):
        classifier = EmotionClassifier(config=decoder_config)
        state = await classifier.classify(_sample_features())

        assert state.dominant_emotion == "neutral"
        assert state.confidence == 0.0
        assert -1.0 <= state.valence <= 1.0
        assert 0.0 <= state.arousal <= 1.0
        assert 0.0 <= state.dominance <= 1.0

    def test_get_model_info_reports_untrained_state(self, decoder_config):
        classifier = EmotionClassifier(config=decoder_config)
        info = classifier.get_model_info()

        assert info["is_trained"] is False


@pytest.mark.unit
class TestMotifDetector:
    """Tests for the multi-label neural motif detector"""

    @pytest.mark.asyncio
    async def test_detect_untrained_returns_no_motifs(self, decoder_config):
        detector = MotifDetector(config=decoder_config)
        motifs = await detector.detect(_sample_features())

        assert motifs == []

    def test_extract_feature_vector_has_expected_length(self, decoder_config):
        detector = MotifDetector(config=decoder_config)
        vector = detector._extract_feature_vector(_sample_features())

        assert vector.shape[0] == detector.input_dim

    def test_generate_motif_labels_returns_known_motif_columns(self, decoder_config):
        detector = MotifDetector(config=decoder_config)
        labels = detector._generate_motif_labels([_sample_features()])

        assert labels.shape == (1, len(SUPPORTED_MOTIFS))
        assert set(detector.supported_motifs) == set(SUPPORTED_MOTIFS)

    def test_summarize_features_returns_band_powers(self, decoder_config):
        detector = MotifDetector(config=decoder_config)
        summary = detector._summarize_features(_sample_features())

        assert "alpha_power" in summary
        assert "frontal_asymmetry" in summary
        assert "artifact_ratio" in summary


@pytest.mark.unit
class TestSyntheticDataGenerator:
    """Tests for synthetic training data generation"""

    @pytest.mark.asyncio
    async def test_generate_training_data_shapes(self):
        generator = SyntheticDataGenerator(seed=42)
        training_data = await generator.generate_training_data(n_samples=5, n_channels=8, embedding_dim=16)

        assert len(training_data.features) == 5
        assert len(training_data.targets) == 5
        assert len(training_data.session_ids) == 5
        assert training_data.metadata["n_channels"] == 8

    @pytest.mark.asyncio
    async def test_generated_targets_are_unit_embeddings(self):
        generator = SyntheticDataGenerator(seed=42)
        training_data = await generator.generate_training_data(n_samples=3, n_channels=8, embedding_dim=16)

        for target in training_data.targets:
            assert len(target) == 16
            assert np.isclose(np.linalg.norm(target), 1.0, atol=1e-6)

    @pytest.mark.asyncio
    async def test_generated_targets_vary_with_features(self):
        generator = SyntheticDataGenerator(seed=42)
        training_data = await generator.generate_training_data(n_samples=20, n_channels=8, embedding_dim=16)

        unique_targets = {tuple(round(v, 8) for v in target) for target in training_data.targets}
        assert len(unique_targets) == 20

    @pytest.mark.asyncio
    async def test_generated_features_have_expected_keys(self):
        generator = SyntheticDataGenerator(seed=42)
        training_data = await generator.generate_training_data(n_samples=1, n_channels=8)
        feature_sample = training_data.features[0]

        for key in ("delta_power", "alpha_power", "hjorth_activity", "frontal_asymmetry", "artifact_ratio"):
            assert key in feature_sample

        assert len(feature_sample["alpha_power"]) == 8
        assert 0.0 <= feature_sample["artifact_ratio"] <= 1.0

    def test_seeded_generator_is_reproducible(self):
        gen_a = SyntheticDataGenerator(seed=123)
        gen_b = SyntheticDataGenerator(seed=123)

        sample_a = gen_a._generate_feature_sample(n_channels=4)
        sample_b = gen_b._generate_feature_sample(n_channels=4)

        assert sample_a["alpha_power"] == sample_b["alpha_power"]


@pytest.mark.unit
class TestTrainingPipeline:
    """End-to-end smoke tests for synthetic training and checkpoint save/load"""

    @pytest.mark.asyncio
    async def test_eeg_to_clip_train_save_load_round_trip(self, decoder_config, tmp_path):
        generator = SyntheticDataGenerator(seed=1)
        training_data = await generator.generate_training_data(n_samples=40, n_channels=8, embedding_dim=32)

        decoder = EEGToCLIPDecoder(config=decoder_config)
        decoder.output_dim = 32
        await decoder.train_synthetic(training_data)
        assert decoder.is_trained is True

        checkpoint_path = tmp_path / "eeg_to_clip.pth"
        await decoder.save_model(str(checkpoint_path))

        reloaded = EEGToCLIPDecoder(config=decoder_config)
        await reloaded.load_model(str(checkpoint_path))
        assert reloaded.is_trained is True

        embedding = await reloaded.decode(_sample_features())
        assert embedding.shape == (32,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-4)

    @pytest.mark.asyncio
    async def test_emotion_classifier_train_save_load_round_trip(self, decoder_config, tmp_path):
        generator = SyntheticDataGenerator(seed=2)
        training_data = await generator.generate_training_data(n_samples=40, n_channels=8)

        classifier = EmotionClassifier(config=decoder_config)
        await classifier.train_synthetic(training_data)
        assert classifier.is_trained is True

        checkpoint_path = tmp_path / "emotion_classifier.pth"
        await classifier.save_model(str(checkpoint_path))

        reloaded = EmotionClassifier(config=decoder_config)
        await reloaded.load_model(str(checkpoint_path))
        assert reloaded.is_trained is True

        state = await reloaded.classify(_sample_features())
        assert state.dominant_emotion in reloaded.supported_emotions

    @pytest.mark.asyncio
    async def test_motif_detector_train_save_load_round_trip(self, decoder_config, tmp_path):
        generator = SyntheticDataGenerator(seed=3)
        training_data = await generator.generate_training_data(n_samples=40, n_channels=8)

        detector = MotifDetector(config=decoder_config)
        await detector.train_synthetic(training_data)
        assert detector.is_trained is True

        checkpoint_path = tmp_path / "motif_detector.pth"
        await detector.save_model(str(checkpoint_path))

        reloaded = MotifDetector(config=decoder_config)
        await reloaded.load_model(str(checkpoint_path))
        assert reloaded.is_trained is True

        motifs = await reloaded.detect(_sample_features())
        assert all(motif.motif_type in SUPPORTED_MOTIFS for motif in motifs)
