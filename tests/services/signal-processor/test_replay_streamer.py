"""
Unit tests for ReplayStreamer and fMRIProcessor against real recorded data
from datasets/real/ (PhysioNet EEG and OpenNeuro fMRI samples).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add service to path (hyphenated directory can't be imported as a package)
service_path = Path(__file__).parent.parent.parent.parent / "services" / "signal-processor"
sys.path.insert(0, str(service_path))

from streamers.replay_streamer import ReplayStreamer  # noqa: E402
from processors.fmri_processor import fMRIProcessor  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent.parent.parent
EEG_FILE = REPO_ROOT / "datasets" / "real" / "eeg" / "S001R01.edf"
FMRI_FILE = REPO_ROOT / "datasets" / "real" / "fmri" / "sub-01_inplaneT2.nii.gz"


@pytest.mark.unit
class TestReplayStreamer:
    """Tests for replaying real recorded EEG data"""

    @pytest.mark.asyncio
    async def test_stream_yields_real_eeg_windows(self):
        streamer = ReplayStreamer(config={
            "file_path": str(EEG_FILE),
            "window_size": 1.0,
            "update_rate": 1000,
        })

        gen = streamer.stream()
        window = await gen.__anext__()
        await gen.aclose()

        assert window.channels == ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
        assert window.sampling_rate == 160
        assert window.data.shape == (8, 160)
        assert window.metadata["source"] == "replay_streamer"
        assert window.metadata["file"] == "S001R01.edf"

    @pytest.mark.asyncio
    async def test_stream_loops_at_end_of_recording(self):
        streamer = ReplayStreamer(config={
            "file_path": str(EEG_FILE),
            "window_size": 1.0,
            "update_rate": 1000,
            "loop": True,
        })

        gen = streamer.stream()

        # Pull enough windows to wrap past the end of the recording
        windows = []
        for _ in range(70):
            windows.append(await gen.__anext__())
        await gen.aclose()

        assert all(w.data.shape == (8, 160) for w in windows)
        assert streamer._data is not None
        total_samples = streamer._data.shape[1]
        assert sum(w.data.shape[1] for w in windows) > total_samples

    @pytest.mark.asyncio
    async def test_amplitudes_are_in_microvolt_range(self):
        streamer = ReplayStreamer(config={
            "file_path": str(EEG_FILE),
            "window_size": 1.0,
            "update_rate": 1000,
        })

        gen = streamer.stream()
        window = await gen.__anext__()
        await gen.aclose()

        # Real EEG amplitudes are roughly tens to a few hundred microvolts
        assert np.max(np.abs(window.data)) < 1000
        assert np.max(np.abs(window.data)) > 1


@pytest.mark.unit
class TestFMRIProcessorRealData:
    """Tests for processing a real NIfTI fMRI volume"""

    @pytest.mark.asyncio
    async def test_process_real_fmri_volume(self):
        processor = fMRIProcessor()
        result = await processor.process_file(str(FMRI_FILE), n_rois=8)

        assert result.data.shape[0] == 8
        assert result.channels == [f"roi_{i}" for i in range(8)]
        assert result.metadata["source"] == "fmri_processor"
        assert result.metadata["original_shape"][:3] == [128, 128, 33]
