"""
Unit tests for EEG signal processing: filtering, artifact detection,
quality metrics, and neural feature extraction.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add service to path (hyphenated directory can't be imported as a package)
service_path = Path(__file__).parent.parent.parent.parent / "services" / "signal-processor"
sys.path.insert(0, str(service_path))

from models.signal_models import EEGConfig, ProcessingConfig, SignalData  # noqa: E402
from processors.eeg_processor import EEGProcessor  # noqa: E402
from feature_extractors.neural_features import NeuralFeatureExtractor  # noqa: E402


@pytest.fixture
def eeg_config():
    return EEGConfig(sampling_rate=250, channels=["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"])


@pytest.fixture
def synthetic_signal(eeg_config):
    rng = np.random.default_rng(42)
    data = rng.standard_normal((len(eeg_config.channels), 1000))
    return SignalData(data=data, sampling_rate=eeg_config.sampling_rate, channels=eeg_config.channels)


@pytest.mark.unit
class TestEEGProcessor:
    """Tests for EEGProcessor filtering, referencing, and artifact detection"""

    def test_initialization_uses_default_config(self):
        processor = EEGProcessor()
        assert processor.config.sampling_rate == 250
        assert processor.ica_fitted is False

    def test_apply_bandpass_filter_preserves_shape(self, eeg_config, synthetic_signal):
        processor = EEGProcessor(eeg_config)
        filtered = processor._apply_bandpass_filter(synthetic_signal.data, eeg_config.sampling_rate)

        assert filtered.shape == synthetic_signal.data.shape
        assert np.all(np.isfinite(filtered))

    def test_apply_notch_filter_preserves_shape(self, eeg_config, synthetic_signal):
        processor = EEGProcessor(eeg_config)
        filtered = processor._apply_notch_filter(synthetic_signal.data, eeg_config.sampling_rate)

        assert filtered.shape == synthetic_signal.data.shape

    def test_apply_reference_average_zeroes_channel_mean(self, eeg_config, synthetic_signal):
        processor = EEGProcessor(eeg_config)
        referenced = processor._apply_reference(synthetic_signal.data)

        assert np.allclose(np.mean(referenced, axis=0), 0, atol=1e-6)

    def test_apply_reference_none_returns_original(self, synthetic_signal):
        config = EEGConfig(channels=synthetic_signal.channels, reference="none")
        processor = EEGProcessor(config)
        referenced = processor._apply_reference(synthetic_signal.data)

        np.testing.assert_array_equal(referenced, synthetic_signal.data)

    @pytest.mark.asyncio
    async def test_process_preserves_shape_and_marks_metadata(self, eeg_config, synthetic_signal):
        processor = EEGProcessor(eeg_config)
        processed = await processor.process(synthetic_signal)

        assert processed.data.shape == synthetic_signal.data.shape
        assert processed.channels == synthetic_signal.channels
        assert processed.metadata["processing"]["bandpass_applied"] is True
        assert processed.metadata["processing"]["reference"] == eeg_config.reference

    def test_detect_artifacts_returns_expected_keys(self, eeg_config, synthetic_signal):
        processor = EEGProcessor(eeg_config)
        artifacts = processor.detect_artifacts(synthetic_signal.data, eeg_config.sampling_rate)

        assert set(artifacts.keys()) == {"eye_blinks", "muscle_artifacts", "flat_channels", "bad_channels"}
        assert isinstance(artifacts["bad_channels"], list)
        assert artifacts["flat_channels"] == 0

    def test_detect_artifacts_flags_flat_channel(self, eeg_config, synthetic_signal):
        processor = EEGProcessor(eeg_config)
        data = synthetic_signal.data.copy()
        data[0, :] = 0.0  # make channel 0 perfectly flat

        artifacts = processor.detect_artifacts(data, eeg_config.sampling_rate)

        assert artifacts["flat_channels"] >= 1
        assert 0 in artifacts["bad_channels"]

    def test_get_signal_quality_metrics_returns_expected_keys(self, eeg_config, synthetic_signal):
        processor = EEGProcessor(eeg_config)
        metrics = processor.get_signal_quality_metrics(synthetic_signal.data)

        for key in ("snr_db", "channel_correlation", "amplitude_range", "alpha_ratio"):
            assert key in metrics
            assert np.isfinite(metrics[key])


@pytest.mark.unit
class TestNeuralFeatureExtractor:
    """Tests for spectral, temporal, connectivity, and asymmetry feature extraction"""

    @pytest.mark.asyncio
    async def test_extract_returns_features_with_correct_lengths(self, eeg_config, synthetic_signal):
        extractor = NeuralFeatureExtractor(ProcessingConfig())
        features = await extractor.extract(synthetic_signal)

        n_channels = len(eeg_config.channels)
        for band_powers in (
            features.delta_power,
            features.theta_power,
            features.alpha_power,
            features.beta_power,
            features.gamma_power,
        ):
            assert len(band_powers) == n_channels

        assert len(features.hjorth_activity) == n_channels
        assert len(features.hjorth_mobility) == n_channels
        assert len(features.hjorth_complexity) == n_channels
        assert 0.0 <= features.artifact_ratio <= 1.0

    @pytest.mark.asyncio
    async def test_extract_connectivity_features_have_correct_shape(self, eeg_config, synthetic_signal):
        extractor = NeuralFeatureExtractor(ProcessingConfig())
        features = await extractor.extract(synthetic_signal)

        n_channels = len(eeg_config.channels)
        assert features.coherence_matrix.shape == (n_channels, n_channels)
        assert features.phase_lag_index.shape == (n_channels, n_channels)

    @pytest.mark.asyncio
    async def test_extract_batch_returns_dataframe(self, synthetic_signal):
        extractor = NeuralFeatureExtractor(ProcessingConfig())
        df = await extractor.extract_batch(synthetic_signal)

        assert len(df) == 1
        assert "alpha_power" in df.columns

    def test_create_minimal_features_on_failure(self):
        extractor = NeuralFeatureExtractor(ProcessingConfig())
        minimal = extractor._create_minimal_features(n_channels=8)

        assert len(minimal.delta_power) == 8
        assert minimal.artifact_ratio == 1.0
