"""
Neural Motif Detector

Detects neural patterns/motifs (e.g. meditation, stress, focus) from
extracted neural features using a multi-label neural network.
"""

import asyncio
import os
import uuid

import structlog
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from models.decoder_models import NeuralMotif, DecoderConfig, TrainingData


SUPPORTED_MOTIFS = [
    'meditation', 'stress', 'focus', 'creativity', 'relaxation',
    'alertness', 'fatigue', 'confusion', 'flow_state', 'anxiety'
]


class MotifDataset(Dataset):
    """Dataset for multi-label neural motif detection"""

    def __init__(self, features: List[Dict[str, Any]], labels: np.ndarray,
                 scaler: Optional[StandardScaler] = None):
        self.features = features
        self.labels = labels
        self.scaler = scaler

        self.feature_vectors = self._extract_feature_vectors()

        if self.scaler is not None:
            self.feature_vectors = self.scaler.transform(self.feature_vectors)

    def _extract_feature_vectors(self) -> np.ndarray:
        """Extract feature vectors from nested dictionaries (shared layout with emotion classifier)"""
        vectors = []
        for feature_dict in self.features:
            vector = []

            for band in ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']:
                if band in feature_dict:
                    vector.extend(feature_dict[band])
                else:
                    vector.extend([0.0] * 8)

            for param in ['hjorth_activity', 'hjorth_mobility', 'hjorth_complexity']:
                if param in feature_dict:
                    vector.extend(feature_dict[param])
                else:
                    vector.extend([0.0] * 8)

            vector.append(feature_dict.get('frontal_asymmetry', 0.0))
            vector.append(feature_dict.get('parietal_asymmetry', 0.0))
            vector.append(feature_dict.get('artifact_ratio', 0.0))
            vector.append(float(feature_dict.get('eye_blink_count', 0)))

            for stat in ['mean_amplitude', 'std_amplitude', 'skewness', 'kurtosis']:
                if stat in feature_dict:
                    vector.extend(feature_dict[stat])
                else:
                    vector.extend([0.0] * 8)

            vectors.append(vector)

        return np.array(vectors)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_vector = torch.FloatTensor(self.feature_vectors[idx])
        label = torch.FloatTensor(self.labels[idx])
        return feature_vector, label


class MotifDetectorNet(nn.Module):
    """Multi-label neural network for motif detection"""

    def __init__(self, input_dim: int, num_motifs: int, hidden_dims: Optional[List[int]] = None):
        super(MotifDetectorNet, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.motif_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_motifs)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.motif_head(features)
        return torch.sigmoid(logits)


class MotifDetector:
    """Detect neural motifs from processed EEG features"""

    def __init__(self, model_path: Optional[str] = None, config: Optional[DecoderConfig] = None):
        self.model_path = model_path
        self.config = config or DecoderConfig()
        self.logger = structlog.get_logger(__name__)

        self.model = None
        self.scaler = StandardScaler()
        self.device = self._get_device()

        self.input_dim = 100  # Same feature layout as emotion classifier
        self.supported_motifs = list(SUPPORTED_MOTIFS)
        self.is_trained = False

    def _get_device(self) -> torch.device:
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.config.device)

    async def load_model(self, model_path: Optional[str] = None):
        """Load pre-trained model"""
        try:
            path = model_path or self.model_path
            if path and os.path.exists(path):
                # weights_only=False: checkpoints bundle a fitted StandardScaler alongside
                # the tensor weights, and we only ever load checkpoints this codebase trained.
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)

                self.model = MotifDetectorNet(
                    input_dim=checkpoint.get('input_dim', self.input_dim),
                    num_motifs=len(self.supported_motifs),
                    hidden_dims=checkpoint.get('hidden_dims', [256, 128, 64])
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()

                if 'scaler' in checkpoint:
                    self.scaler = checkpoint['scaler']
                if 'supported_motifs' in checkpoint:
                    self.supported_motifs = checkpoint['supported_motifs']

                self.is_trained = True
                self.logger.info("Motif detector loaded successfully", model_path=path)

            else:
                self.logger.warning("Model file not found, will train from scratch", path=path)

        except Exception as e:
            self.logger.error("Failed to load motif detector", error=str(e))
            raise

    async def save_model(self, model_path: str):
        """Save trained model"""
        try:
            if self.model is None:
                raise ValueError("No model to save")

            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'input_dim': self.input_dim,
                'hidden_dims': [256, 128, 64],
                'scaler': self.scaler,
                'supported_motifs': self.supported_motifs,
                'config': self.config.dict()
            }

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(checkpoint, model_path)

            self.logger.info("Motif detector saved successfully", model_path=model_path)

        except Exception as e:
            self.logger.error("Failed to save motif detector", error=str(e))
            raise

    async def detect(self, features: Dict[str, Any]) -> List[NeuralMotif]:
        """Detect active neural motifs from features"""
        try:
            if not self.is_trained or self.model is None:
                self.logger.warning("Model not trained, returning no motifs")
                return []

            feature_vector = self._extract_feature_vector(features)

            if hasattr(self.scaler, 'mean_'):
                feature_vector = self.scaler.transform([feature_vector])[0]

            feature_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)

            with torch.no_grad():
                probs = self.model(feature_tensor).cpu().numpy()[0]

            motifs = []
            for motif_type, confidence in zip(self.supported_motifs, probs):
                confidence = float(confidence)
                if confidence >= self.config.min_confidence_threshold:
                    motifs.append(NeuralMotif(
                        motif_id=str(uuid.uuid4()),
                        motif_type=motif_type,
                        confidence=confidence,
                        features=self._summarize_features(features),
                        description=f"Detected '{motif_type}' pattern with {confidence:.2f} confidence"
                    ))

            return motifs

        except Exception as e:
            self.logger.error("Motif detection failed", error=str(e))
            return []

    def _extract_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from features dictionary (shared layout with emotion classifier)"""
        vector = []

        for band in ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']:
            if band in features:
                vector.extend(features[band])
            else:
                vector.extend([0.0] * 8)

        for param in ['hjorth_activity', 'hjorth_mobility', 'hjorth_complexity']:
            if param in features:
                vector.extend(features[param])
            else:
                vector.extend([0.0] * 8)

        vector.append(features.get('frontal_asymmetry', 0.0))
        vector.append(features.get('parietal_asymmetry', 0.0))
        vector.append(features.get('artifact_ratio', 0.0))
        vector.append(float(features.get('eye_blink_count', 0)))

        for stat in ['mean_amplitude', 'std_amplitude', 'skewness', 'kurtosis']:
            if stat in features:
                vector.extend(features[stat])
            else:
                vector.extend([0.0] * 8)

        return np.array(vector)

    def _summarize_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Summarize key band powers that drove a motif detection"""
        summary = {}
        for band in ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']:
            values = features.get(band, [])
            summary[band] = float(np.mean(values)) if len(values) else 0.0
        summary['frontal_asymmetry'] = float(features.get('frontal_asymmetry', 0.0))
        summary['artifact_ratio'] = float(features.get('artifact_ratio', 0.0))
        return summary

    async def train_synthetic(self, training_data: TrainingData):
        """Train the motif detector with synthetic data"""
        try:
            self.logger.info("Starting motif detector synthetic training", n_samples=len(training_data.features))

            features = training_data.features
            labels = self._generate_motif_labels(features)

            X_train, X_val, y_train, y_val = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )

            train_dataset = MotifDataset(X_train, y_train, scaler=None)
            self.scaler.fit(train_dataset.feature_vectors)

            train_dataset = MotifDataset(X_train, y_train, scaler=self.scaler)
            val_dataset = MotifDataset(X_val, y_val, scaler=self.scaler)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            self.model = MotifDetectorNet(
                input_dim=train_dataset.feature_vectors.shape[1],
                num_motifs=len(self.supported_motifs)
            ).to(self.device)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0

            for epoch in range(100):
                train_loss = await self._train_epoch(train_loader, optimizer)
                val_loss = await self._validate_epoch(val_loader)
                scheduler.step(val_loss)

                self.logger.info(
                    "Motif detector epoch completed",
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    self.logger.info("Early stopping triggered", epoch=epoch)
                    break

            self.is_trained = True
            self.logger.info("Motif detector training completed", best_val_loss=best_val_loss)

        except Exception as e:
            self.logger.error("Motif detector synthetic training failed", error=str(e))
            raise

    def _generate_motif_labels(self, features: List[Dict[str, Any]]) -> np.ndarray:
        """Generate multi-hot motif labels from synthetic features using simple heuristics"""
        labels = np.zeros((len(features), len(self.supported_motifs)), dtype=np.float32)
        motif_idx = {motif: i for i, motif in enumerate(self.supported_motifs)}

        for i, feature_dict in enumerate(features):
            alpha = np.mean(feature_dict.get('alpha_power', [0.5] * 8))
            beta = np.mean(feature_dict.get('beta_power', [0.5] * 8))
            theta = np.mean(feature_dict.get('theta_power', [0.5] * 8))
            gamma = np.mean(feature_dict.get('gamma_power', [0.5] * 8))
            delta = np.mean(feature_dict.get('delta_power', [0.5] * 8))
            asymmetry = feature_dict.get('frontal_asymmetry', 0.0)
            artifact_ratio = feature_dict.get('artifact_ratio', 0.0)

            if alpha > 1.2 and theta > 1.0 and beta < 0.8:
                labels[i, motif_idx['meditation']] = 1.0
            if beta > 1.5 and gamma > 1.0 and asymmetry < -0.2:
                labels[i, motif_idx['stress']] = 1.0
            if beta > 1.2 and alpha < 0.8:
                labels[i, motif_idx['focus']] = 1.0
            if theta > 1.0 and alpha > 0.8 and asymmetry > 0.1:
                labels[i, motif_idx['creativity']] = 1.0
            if alpha > 1.2 and beta < 0.6:
                labels[i, motif_idx['relaxation']] = 1.0
            if beta > 1.0 and gamma > 0.8 and artifact_ratio < 0.2:
                labels[i, motif_idx['alertness']] = 1.0
            if delta > 1.2 and beta < 0.5:
                labels[i, motif_idx['fatigue']] = 1.0
            if theta > 1.2 and 0.6 < beta < 1.2:
                labels[i, motif_idx['confusion']] = 1.0
            if 0.8 < alpha < 1.3 and 0.8 < beta < 1.3 and artifact_ratio < 0.15:
                labels[i, motif_idx['flow_state']] = 1.0
            if beta > 1.5 and gamma > 1.2 and artifact_ratio > 0.3:
                labels[i, motif_idx['anxiety']] = 1.0

        return labels

    async def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_features)
            loss = F.binary_cross_entropy(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    async def _validate_epoch(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_features)
                loss = F.binary_cross_entropy(outputs, batch_labels)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "is_trained": self.is_trained,
            "input_dim": self.input_dim,
            "num_motifs": len(self.supported_motifs),
            "supported_motifs": self.supported_motifs,
            "device": str(self.device),
            "model_path": self.model_path,
            "scaler_fitted": hasattr(self.scaler, 'mean_')
        }
