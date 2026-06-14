"""
Replay EEG Streamer

Streams real, pre-recorded EEG data from an EDF file into SignalData
windows for the processing pipeline, looping the recording for
continuous playback.
"""

import asyncio
import logging
import os
from typing import Any, AsyncGenerator, Dict, Optional

import numpy as np

from models.signal_models import EEGConfig, SignalData

DEFAULT_REPLAY_FILE = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "datasets", "real", "eeg", "S001R01.edf",
    )
)


class ReplayStreamer:
    """Replay a real recorded EEG file as a continuous signal stream"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        self.file_path = self.config.get("file_path", DEFAULT_REPLAY_FILE)
        self.window_size = self.config.get("window_size", 4.0)  # seconds
        self.update_rate = self.config.get("update_rate", 10)  # Hz
        self.loop_playback = self.config.get("loop", True)

        self.eeg_config = EEGConfig(
            channels=self.config.get("channels", [
                "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"
            ])
        )

        self._data: Optional[np.ndarray] = None
        self._position = 0
        self._window_index = 0

    def _load(self) -> None:
        """Load the EDF recording and select the configured channels"""
        import mne

        raw = mne.io.read_raw_edf(self.file_path, preload=True, verbose="ERROR")

        # eegmmidb channel names use trailing dots (e.g. "Fp1.")
        raw.rename_channels({ch: ch.rstrip(".") for ch in raw.ch_names})

        available = [ch for ch in self.eeg_config.channels if ch in raw.ch_names]
        if not available:
            available = raw.ch_names[: len(self.eeg_config.channels)]

        raw.pick(available)
        self.eeg_config.channels = available
        self.eeg_config.sampling_rate = int(raw.info["sfreq"])

        # mne returns volts; convert to microvolts to match MockStreamer conventions
        self._data = raw.get_data() * 1e6

    async def stream(self) -> AsyncGenerator[SignalData, None]:
        """Yield windows of real recorded EEG data, looping at the end"""
        if self._data is None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load)

        window_samples = max(1, int(self.window_size * self.eeg_config.sampling_rate))
        n_total_samples = self._data.shape[1]

        try:
            while True:
                start = self._position
                end = start + window_samples

                if end <= n_total_samples:
                    window = self._data[:, start:end]
                else:
                    if not self.loop_playback:
                        break
                    tail = self._data[:, start:n_total_samples]
                    remaining = window_samples - tail.shape[1]
                    head = self._data[:, :remaining]
                    window = np.concatenate([tail, head], axis=1)

                self._position = end % n_total_samples

                yield SignalData(
                    data=window,
                    sampling_rate=self.eeg_config.sampling_rate,
                    channels=self.eeg_config.channels,
                    metadata={
                        "source": "replay_streamer",
                        "file": os.path.basename(self.file_path),
                        "window_index": self._window_index,
                    },
                )

                self._window_index += 1
                await asyncio.sleep(1.0 / self.update_rate)

        except Exception as e:
            self.logger.error(f"Error in replay stream: {e}")
