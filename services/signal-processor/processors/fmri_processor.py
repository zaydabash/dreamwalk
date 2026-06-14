"""
fMRI Signal Processor

Loads fMRI volumes (NIfTI) and reduces them to region-averaged time series
suitable for the shared neural feature extraction pipeline.
"""

import asyncio
import logging
from typing import List, Optional

import numpy as np

from models.signal_models import SignalData, fMRIConfig


class fMRIProcessor:
    """Process fMRI data files into SignalData windows"""

    def __init__(self, config: Optional[fMRIConfig] = None):
        self.config = config or fMRIConfig()
        self.logger = logging.getLogger(__name__)

    async def process_file(self, file_path: str, n_rois: int = 16) -> SignalData:
        """Load an fMRI volume and reduce it to ROI-averaged time series"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process_file_sync, file_path, n_rois)

    def _process_file_sync(self, file_path: str, n_rois: int) -> SignalData:
        try:
            import nibabel as nib

            img = nib.load(file_path)
            data = img.get_fdata()
        except Exception as e:
            self.logger.error(f"Failed to load fMRI file {file_path}: {e}")
            raise

        if data.ndim == 3:
            # Single 3D volume, treat as a single time point
            data = data[..., np.newaxis]

        if data.ndim != 4:
            raise ValueError(f"Expected 3D or 4D fMRI data, got shape {data.shape}")

        roi_timeseries, channels = self._parcellate(data, n_rois)

        sampling_rate = 1.0 / self.config.tr if self.config.tr > 0 else 0.5

        return SignalData(
            data=roi_timeseries,
            sampling_rate=sampling_rate,
            channels=channels,
            metadata={
                "source": "fmri_processor",
                "file_path": file_path,
                "original_shape": list(data.shape),
                "tr": self.config.tr,
                "n_rois": n_rois,
            },
        )

    def _parcellate(self, data: np.ndarray, n_rois: int) -> "tuple[np.ndarray, List[str]]":
        """Reduce a 4D fMRI volume to n_rois average time series via a coarse spatial grid"""
        x, y, z, t = data.shape

        # Flatten spatial dimensions and split into roughly equal blocks
        voxels = data.reshape(x * y * z, t)
        n_voxels = voxels.shape[0]
        n_rois = max(1, min(n_rois, n_voxels))

        boundaries = np.linspace(0, n_voxels, n_rois + 1, dtype=int)
        roi_timeseries = np.zeros((n_rois, t))

        for i in range(n_rois):
            start, end = boundaries[i], boundaries[i + 1]
            block = voxels[start:end]
            # Ignore voxels that are entirely zero (background/outside brain)
            valid_mask = np.any(block != 0, axis=1)
            if np.any(valid_mask):
                roi_timeseries[i] = block[valid_mask].mean(axis=0)
            else:
                roi_timeseries[i] = block.mean(axis=0)

        channels = [f"roi_{i}" for i in range(n_rois)]
        return roi_timeseries, channels
