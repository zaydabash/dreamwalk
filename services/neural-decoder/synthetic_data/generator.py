"""
Synthetic Training Data Generator

Generates synthetic neural feature samples for training the EEG-to-CLIP
decoder, emotion classifier, and motif detector without requiring real
recorded EEG data.
"""

import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import structlog
from models.decoder_models import TrainingData


class SyntheticDataGenerator:
    """Generate synthetic neural feature data for training decoder models"""

    def __init__(self, seed: Optional[int] = None):
        self.logger = structlog.get_logger(__name__)
        self.rng = np.random.default_rng(seed)
        self._projection: Optional[np.ndarray] = None

    async def generate_training_data(
        self, n_samples: int = 10000, n_channels: int = 8, embedding_dim: int = 512
    ) -> TrainingData:
        """Generate a synthetic training dataset of neural feature samples"""
        try:
            features = [self._generate_feature_sample(n_channels) for _ in range(n_samples)]
            targets = [
                self._generate_target_embedding(sample, embedding_dim) for sample in features
            ]
            session_ids = [str(uuid.uuid4()) for _ in range(n_samples)]

            return TrainingData(
                features=features,
                targets=targets,
                metadata={
                    "n_samples": n_samples,
                    "n_channels": n_channels,
                    "embedding_dim": embedding_dim,
                    "generator": "SyntheticDataGenerator",
                },
                session_ids=session_ids,
            )

        except Exception as e:
            self.logger.error("Synthetic data generation failed", error=str(e))
            raise

    def _generate_target_embedding(
        self, feature_sample: Dict[str, Any], embedding_dim: int
    ) -> List[float]:
        """Derive a CLIP-style unit embedding from a feature sample's band/asymmetry summary.

        Uses a fixed random projection so the embedding is a learnable function of the
        input features (not pure noise), giving the EEG-to-CLIP decoder a real signal to fit.
        """
        latent = np.array(
            [
                np.mean(feature_sample["delta_power"]),
                np.mean(feature_sample["theta_power"]),
                np.mean(feature_sample["alpha_power"]),
                np.mean(feature_sample["beta_power"]),
                np.mean(feature_sample["gamma_power"]),
                feature_sample["frontal_asymmetry"],
                feature_sample["parietal_asymmetry"],
                feature_sample["artifact_ratio"],
            ]
        )

        if self._projection is None or self._projection.shape != (latent.shape[0], embedding_dim):
            self._projection = self.rng.normal(
                scale=1.0 / np.sqrt(latent.shape[0]), size=(latent.shape[0], embedding_dim)
            )

        embedding = latent @ self._projection
        embedding += self.rng.normal(scale=0.05, size=embedding_dim)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    def _generate_feature_sample(self, n_channels: int) -> Dict[str, Any]:
        """Generate a single synthetic neural feature dictionary"""
        band_powers = {
            "delta_power": self._band_power(n_channels),
            "theta_power": self._band_power(n_channels),
            "alpha_power": self._band_power(n_channels),
            "beta_power": self._band_power(n_channels),
            "gamma_power": self._band_power(n_channels),
        }

        hjorth = {
            "hjorth_activity": self.rng.lognormal(mean=0.0, sigma=0.5, size=n_channels).tolist(),
            "hjorth_mobility": self.rng.lognormal(mean=-0.5, sigma=0.4, size=n_channels).tolist(),
            "hjorth_complexity": self.rng.lognormal(mean=0.0, sigma=0.3, size=n_channels).tolist(),
        }

        summary_stats = {
            "mean_amplitude": self.rng.normal(loc=0.0, scale=1.0, size=n_channels).tolist(),
            "std_amplitude": self.rng.lognormal(mean=0.0, sigma=0.3, size=n_channels).tolist(),
            "skewness": self.rng.normal(loc=0.0, scale=1.0, size=n_channels).tolist(),
            "kurtosis": self.rng.normal(loc=0.0, scale=3.0, size=n_channels).tolist(),
        }

        return {
            **band_powers,
            **hjorth,
            "frontal_asymmetry": float(self.rng.normal(loc=0.0, scale=0.4)),
            "parietal_asymmetry": float(self.rng.normal(loc=0.0, scale=0.4)),
            "artifact_ratio": float(np.clip(self.rng.exponential(scale=0.1), 0.0, 1.0)),
            "eye_blink_count": int(self.rng.poisson(lam=2)),
            **summary_stats,
        }

    def _band_power(self, n_channels: int) -> List[float]:
        """Sample band power values centered around 1.0 with realistic spread"""
        return self.rng.lognormal(mean=0.0, sigma=0.4, size=n_channels).tolist()
