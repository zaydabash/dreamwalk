"""
LSL EEG Streamer

Streams real EEG data from a LabStreamingLayer (LSL) source into SignalData
windows for the processing pipeline.
"""

import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional

import numpy as np

from models.signal_models import SignalData, EEGConfig


class LSLStreamer:
    """Stream EEG data from a LabStreamingLayer source"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        self.stream_type = self.config.get("stream_type", "EEG")
        self.resolve_timeout = self.config.get("resolve_timeout", 5.0)
        self.window_size = self.config.get("window_size", 4.0)  # seconds

        self.eeg_config = EEGConfig(
            sampling_rate=self.config.get("sampling_rate", 250),
            channels=self.config.get("channels", [
                "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"
            ]),
        )

        self._inlet = None

    async def _connect(self):
        """Resolve and connect to an LSL stream"""
        from pylsl import StreamInlet, resolve_byprop

        loop = asyncio.get_event_loop()
        streams = await loop.run_in_executor(
            None, resolve_byprop, "type", self.stream_type, 1, self.resolve_timeout
        )

        if not streams:
            raise RuntimeError(f"No LSL stream found of type '{self.stream_type}'")

        self._inlet = StreamInlet(streams[0])

        info = self._inlet.info()
        sampling_rate = info.nominal_srate()
        if sampling_rate and sampling_rate > 0:
            self.eeg_config.sampling_rate = int(sampling_rate)

        n_channels = info.channel_count()
        channels = self._extract_channel_labels(info, n_channels)
        if channels:
            self.eeg_config.channels = channels

        self.logger.info(
            f"Connected to LSL stream '{info.name()}' "
            f"({n_channels} channels @ {self.eeg_config.sampling_rate} Hz)"
        )

    @staticmethod
    def _extract_channel_labels(info, n_channels: int) -> List[str]:
        """Best-effort extraction of channel labels from LSL stream metadata"""
        try:
            channels = []
            desc = info.desc()
            channels_node = desc.child("channels")
            channel_node = channels_node.child("channel")
            while channel_node.name() == "channel" and not channel_node.empty():
                label = channel_node.child_value("label")
                if label:
                    channels.append(label)
                channel_node = channel_node.next_sibling()

            if len(channels) == n_channels:
                return channels
        except Exception:
            pass

        return [f"ch{i}" for i in range(n_channels)]

    async def stream(self) -> AsyncGenerator[SignalData, None]:
        """Yield windows of EEG data pulled from the LSL inlet"""
        if self._inlet is None:
            await self._connect()

        loop = asyncio.get_event_loop()
        window_samples = max(1, int(self.window_size * self.eeg_config.sampling_rate))
        n_channels = len(self.eeg_config.channels)
        buffer: List[List[float]] = []

        try:
            while True:
                chunk, _timestamps = await loop.run_in_executor(
                    None, self._inlet.pull_chunk, 1.0, window_samples
                )

                if chunk:
                    buffer.extend(chunk)

                if len(buffer) >= window_samples:
                    window = np.array(buffer[:window_samples], dtype=float).T

                    if window.shape[0] != n_channels:
                        n_channels = window.shape[0]
                        if len(self.eeg_config.channels) != n_channels:
                            self.eeg_config.channels = [f"ch{i}" for i in range(n_channels)]

                    yield SignalData(
                        data=window,
                        sampling_rate=self.eeg_config.sampling_rate,
                        channels=self.eeg_config.channels,
                        metadata={"source": "lsl_streamer"},
                    )

                    buffer = buffer[window_samples:]
                else:
                    await asyncio.sleep(0.05)

        except Exception as e:
            self.logger.error(f"Error in LSL stream: {e}")
