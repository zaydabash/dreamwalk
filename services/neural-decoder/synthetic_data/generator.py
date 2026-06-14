"""
Synthetic Training Data Generator

Generates synthetic neural feature samples for training the EEG-to-CLIP
decoder, emotion classifier, and motif detector without requiring real
recorded EEG data.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from models.decoder_models import TrainingData


class SyntheticDataGenerator:
    """Generate synthetic neural feature data for training decoder models"""

    def __init__(self, seed: Optional[int] = None):
        self.logger = logging.getLogger(__name__)
        self.rng = np.random.default_rng(seed)

    async def generate_training_data(self, n_samples: int = 10000, n_channels: int = 8) -> TrainingData:
        """Generate a synthetic training dataset of neural feature samples"""
        try:
            features = [self._generate_feature_sample(n_channels) for _ in range(n_samples)]
            targets = [{} for _ in range(n_samples)]
            session_ids = [str(uuid.uuid4()) for _ in range(n_samples)]

            return TrainingData(
                features=features,
                targets=targets,
                metadata={
                    "n_samples": n_samples,
                    "n_channels": n_channels,
                    "generator": "SyntheticDataGenerator",
                },
                session_ids=session_ids,
            )

        except Exception as e:
            self.logger.error("Synthetic data generation failed", error=str(e))
            raise

    def _generate_feature_sample(self, n_channels: int) -> Dict[str, Any]:
        """Generate a single synthetic neural feature dictionary"""
        band_powers = {
            'delta_power': self._band_power(n_channels),
            'theta_power': self._band_power(n_channels),
            'alpha_power': self._band_power(n_channels),
            'beta_power': self._band_power(n_channels),
            'gamma_power': self._band_power(n_channels),
        }

        hjorth = {
            'hjorth_activity': self.rng.lognormal(mean=0.0, sigma=0.5, size=n_channels).tolist(),
            'hjorth_mobility': self.rng.lognormal(mean=-0.5, sigma=0.4, size=n_channels).tolist(),
            'hjorth_complexity': self.rng.lognormal(mean=0.0, sigma=0.3, size=n_channels).tolist(),
        }

        summary_stats = {
            'mean_amplitude': self.rng.normal(loc=0.0, scale=1.0, size=n_channels).tolist(),
            'std_amplitude': self.rng.lognormal(mean=0.0, sigma=0.3, size=n_channels).tolist(),
            'skewness': self.rng.normal(loc=0.0, scale=1.0, size=n_channels).tolist(),
            'kurtosis': self.rng.normal(loc=0.0, scale=3.0, size=n_channels).tolist(),
        }

        return {
            **band_powers,
            **hjorth,
            'frontal_asymmetry': float(self.rng.normal(loc=0.0, scale=0.4)),
            'parietal_asymmetry': float(self.rng.normal(loc=0.0, scale=0.4)),
            'artifact_ratio': float(np.clip(self.rng.exponential(scale=0.1), 0.0, 1.0)),
            'eye_blink_count': int(self.rng.poisson(lam=2)),
            **summary_stats,
        }

    def _band_power(self, n_channels: int) -> List[float]:
        """Sample band power values centered around 1.0 with realistic spread"""
        return self.rng.lognormal(mean=0.0, sigma=0.4, size=n_channels).tolist()
