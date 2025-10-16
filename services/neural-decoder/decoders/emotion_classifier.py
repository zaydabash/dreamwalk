"""
Emotion Classifier

Classifies emotional states from neural features.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from ..models.decoder_models import EmotionalState, DecoderConfig, TrainingData


class EmotionDataset(Dataset):
    """Dataset for emotion classification"""
    
    def __init__(self, features: List[Dict[str, Any]], labels: List[str], 
                 scaler: Optional[StandardScaler] = None, label_encoder: Optional[LabelEncoder] = None):
        self.features = features
        self.labels = labels
        self.scaler = scaler
        self.label_encoder = label_encoder
        
        # Extract feature vectors
        self.feature_vectors = self._extract_feature_vectors()
        
        # Encode labels
        if self.label_encoder is not None:
            self.encoded_labels = self.label_encoder.transform(self.labels)
        else:
            self.encoded_labels = np.array([0] * len(self.labels))  # Default encoding
        
        # Scale features
        if self.scaler is not None:
            self.feature_vectors = self.scaler.transform(self.feature_vectors)
    
    def _extract_feature_vectors(self) -> np.ndarray:
        """Extract feature vectors from nested dictionaries"""
        vectors = []
        for feature_dict in self.features:
            vector = []
            
            # Add spectral features
            for band in ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']:
                if band in feature_dict:
                    vector.extend(feature_dict[band])
                else:
                    vector.extend([0.0] * 8)  # Default 8 channels
            
            # Add temporal features
            for param in ['hjorth_activity', 'hjorth_mobility', 'hjorth_complexity']:
                if param in feature_dict:
                    vector.extend(feature_dict[param])
                else:
                    vector.extend([0.0] * 8)
            
            # Add asymmetry features
            vector.append(feature_dict.get('frontal_asymmetry', 0.0))
            vector.append(feature_dict.get('parietal_asymmetry', 0.0))
            
            # Add artifact metrics
            vector.append(feature_dict.get('artifact_ratio', 0.0))
            vector.append(float(feature_dict.get('eye_blink_count', 0)))
            
            # Add summary statistics
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
        label = torch.LongTensor([self.encoded_labels[idx]])
        return feature_vector, label


class EmotionClassifierNet(nn.Module):
    """Neural network for emotion classification"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = None):
        super(EmotionClassifierNet, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        # Build network layers
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
        
        # Output layers for different emotion dimensions
        self.feature_extractor = nn.Sequential(*layers)
        
        # Separate heads for valence, arousal, dominance
        self.valence_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Tanh()  # Valence: -1 to 1
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Arousal: 0 to 1
        )
        
        self.dominance_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Dominance: 0 to 1
        )
        
        # Emotion category classifier
        self.emotion_classifier = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        valence = self.valence_head(features)
        arousal = self.arousal_head(features)
        dominance = self.dominance_head(features)
        emotion_logits = self.emotion_classifier(features)
        
        return {
            'valence': valence,
            'arousal': arousal,
            'dominance': dominance,
            'emotion_logits': emotion_logits
        }


class EmotionClassifier:
    """Emotion classifier from neural features"""
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[DecoderConfig] = None):
        self.model_path = model_path
        self.config = config or DecoderConfig()
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = self._get_device()
        
        # Model parameters
        self.input_dim = 88  # Calculated from feature extraction
        self.supported_emotions = [
            'neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised', 'disgusted',
            'relaxed', 'stressed', 'excited', 'calm', 'focused', 'confused'
        ]
        self.is_trained = False
    
    def _get_device(self) -> torch.device:
        """Get appropriate device for computation"""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.config.device)
    
    async def load_model(self, model_path: Optional[str] = None):
        """Load pre-trained model"""
        try:
            path = model_path or self.model_path
            if path and os.path.exists(path):
                checkpoint = torch.load(path, map_location=self.device)
                
                # Initialize model
                self.model = EmotionClassifierNet(
                    input_dim=checkpoint.get('input_dim', self.input_dim),
                    num_classes=len(self.supported_emotions),
                    hidden_dims=checkpoint.get('hidden_dims', [256, 128, 64])
                )
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                # Load scaler and label encoder
                if 'scaler' in checkpoint:
                    self.scaler = checkpoint['scaler']
                if 'label_encoder' in checkpoint:
                    self.label_encoder = checkpoint['label_encoder']
                
                self.is_trained = True
                self.logger.info("Emotion classifier loaded successfully", model_path=path)
                
            else:
                self.logger.warning("Model file not found, will train from scratch", path=path)
                
        except Exception as e:
            self.logger.error("Failed to load emotion classifier", error=str(e))
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
                'label_encoder': self.label_encoder,
                'supported_emotions': self.supported_emotions,
                'config': self.config.dict()
            }
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(checkpoint, model_path)
            
            self.logger.info("Emotion classifier saved successfully", model_path=model_path)
            
        except Exception as e:
            self.logger.error("Failed to save emotion classifier", error=str(e))
            raise
    
    async def classify(self, features: Dict[str, Any]) -> EmotionalState:
        """Classify emotional state from neural features"""
        try:
            if not self.is_trained or self.model is None:
                # Return neutral state if model not available
                self.logger.warning("Model not trained, returning neutral state")
                return EmotionalState(
                    valence=0.0,
                    arousal=0.5,
                    dominance=0.5,
                    confidence=0.0,
                    dominant_emotion="neutral"
                )
            
            # Extract feature vector
            feature_vector = self._extract_feature_vector(features)
            
            # Normalize if scaler is fitted
            if hasattr(self.scaler, 'mean_'):
                feature_vector = self.scaler.transform([feature_vector])[0]
            
            # Convert to tensor
            feature_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(feature_tensor)
                
                valence = outputs['valence'].cpu().numpy()[0][0]
                arousal = outputs['arousal'].cpu().numpy()[0][0]
                dominance = outputs['dominance'].cpu().numpy()[0][0]
                
                # Get emotion probabilities
                emotion_logits = outputs['emotion_logits'].cpu().numpy()[0]
                emotion_probs = F.softmax(torch.FloatTensor(emotion_logits), dim=0).numpy()
                
                # Get dominant emotion
                dominant_idx = np.argmax(emotion_probs)
                dominant_emotion = self.supported_emotions[dominant_idx]
                confidence = float(emotion_probs[dominant_idx])
                
                # Create emotion scores dictionary
                emotion_scores = {
                    emotion: float(prob) 
                    for emotion, prob in zip(self.supported_emotions, emotion_probs)
                }
            
            return EmotionalState(
                valence=float(valence),
                arousal=float(arousal),
                dominance=float(dominance),
                confidence=confidence,
                dominant_emotion=dominant_emotion,
                emotion_scores=emotion_scores
            )
            
        except Exception as e:
            self.logger.error("Emotion classification failed", error=str(e))
            # Return neutral state on error
            return EmotionalState(
                valence=0.0,
                arousal=0.5,
                dominance=0.5,
                confidence=0.0,
                dominant_emotion="neutral"
            )
    
    def _extract_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from features dictionary"""
        vector = []
        
        # Add spectral features
        for band in ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']:
            if band in features:
                vector.extend(features[band])
            else:
                vector.extend([0.0] * 8)  # Default 8 channels
        
        # Add temporal features
        for param in ['hjorth_activity', 'hjorth_mobility', 'hjorth_complexity']:
            if param in features:
                vector.extend(features[param])
            else:
                vector.extend([0.0] * 8)
        
        # Add asymmetry features
        vector.append(features.get('frontal_asymmetry', 0.0))
        vector.append(features.get('parietal_asymmetry', 0.0))
        
        # Add artifact metrics
        vector.append(features.get('artifact_ratio', 0.0))
        vector.append(float(features.get('eye_blink_count', 0)))
        
        # Add summary statistics
        for stat in ['mean_amplitude', 'std_amplitude', 'skewness', 'kurtosis']:
            if stat in features:
                vector.extend(features[stat])
            else:
                vector.extend([0.0] * 8)
        
        return np.array(vector)
    
    async def train_synthetic(self, training_data: TrainingData):
        """Train model with synthetic data"""
        try:
            self.logger.info("Starting emotion classifier synthetic training", n_samples=len(training_data.features))
            
            # Prepare data - need to create emotion labels from synthetic data
            features = training_data.features
            labels = self._generate_emotion_labels(features)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            
            # Create datasets
            train_dataset = EmotionDataset(X_train, y_train, scaler=None, label_encoder=None)
            val_dataset = EmotionDataset(X_val, y_val, scaler=train_dataset.scaler, 
                                       label_encoder=train_dataset.label_encoder)
            
            # Fit scaler and label encoder on training data
            self.scaler.fit(train_dataset.feature_vectors)
            self.label_encoder.fit(train_dataset.labels)
            
            # Recreate datasets with fitted scaler and encoder
            train_dataset = EmotionDataset(X_train, y_train, scaler=self.scaler, label_encoder=self.label_encoder)
            val_dataset = EmotionDataset(X_val, y_val, scaler=self.scaler, label_encoder=self.label_encoder)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Initialize model
            self.model = EmotionClassifierNet(
                input_dim=train_dataset.feature_vectors.shape[1],
                num_classes=len(self.supported_emotions)
            ).to(self.device)
            
            # Training setup
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            
            # Training loop
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(100):  # Maximum epochs
                # Training
                train_loss = await self._train_epoch(train_loader, optimizer)
                
                # Validation
                val_loss = await self._validate_epoch(val_loader)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                self.logger.info(
                    "Emotion classifier epoch completed",
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss
                )
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    self.logger.info("Early stopping triggered", epoch=epoch)
                    break
            
            self.is_trained = True
            self.logger.info("Emotion classifier training completed", best_val_loss=best_val_loss)
            
        except Exception as e:
            self.logger.error("Emotion classifier synthetic training failed", error=str(e))
            raise
    
    def _generate_emotion_labels(self, features: List[Dict[str, Any]]) -> List[str]:
        """Generate emotion labels from synthetic features"""
        labels = []
        
        for feature_dict in features:
            # Simple rule-based emotion assignment based on spectral features
            alpha_power = np.mean(feature_dict.get('alpha_power', [0.5] * 8))
            beta_power = np.mean(feature_dict.get('beta_power', [0.5] * 8))
            theta_power = np.mean(feature_dict.get('theta_power', [0.5] * 8))
            
            frontal_asymmetry = feature_dict.get('frontal_asymmetry', 0.0)
            
            # Simple emotion mapping
            if alpha_power > 1.2 and beta_power < 0.8:
                if frontal_asymmetry > 0.2:
                    labels.append('relaxed')
                else:
                    labels.append('calm')
            elif beta_power > 1.5:
                if frontal_asymmetry < -0.2:
                    labels.append('stressed')
                else:
                    labels.append('focused')
            elif theta_power > 1.0:
                labels.append('confused')
            else:
                labels.append('neutral')
        
        return labels
    
    async def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.squeeze().to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_features)
            
            # Compute losses
            emotion_loss = F.cross_entropy(outputs['emotion_logits'], batch_labels)
            
            # Simple synthetic targets for regression
            valence_target = torch.randn(batch_features.size(0), 1).to(self.device)
            arousal_target = torch.rand(batch_features.size(0), 1).to(self.device)
            dominance_target = torch.rand(batch_features.size(0), 1).to(self.device)
            
            valence_loss = F.mse_loss(outputs['valence'], valence_target)
            arousal_loss = F.mse_loss(outputs['arousal'], arousal_target)
            dominance_loss = F.mse_loss(outputs['dominance'], dominance_target)
            
            # Combined loss
            loss = emotion_loss + 0.1 * (valence_loss + arousal_loss + dominance_loss)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    async def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.squeeze().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_features)
                
                # Compute losses
                emotion_loss = F.cross_entropy(outputs['emotion_logits'], batch_labels)
                
                # Simple synthetic targets for regression
                valence_target = torch.randn(batch_features.size(0), 1).to(self.device)
                arousal_target = torch.rand(batch_features.size(0), 1).to(self.device)
                dominance_target = torch.rand(batch_features.size(0), 1).to(self.device)
                
                valence_loss = F.mse_loss(outputs['valence'], valence_target)
                arousal_loss = F.mse_loss(outputs['arousal'], arousal_target)
                dominance_loss = F.mse_loss(outputs['dominance'], dominance_target)
                
                # Combined loss
                loss = emotion_loss + 0.1 * (valence_loss + arousal_loss + dominance_loss)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "is_trained": self.is_trained,
            "input_dim": self.input_dim,
            "num_classes": len(self.supported_emotions),
            "supported_emotions": self.supported_emotions,
            "device": str(self.device),
            "model_path": self.model_path,
            "scaler_fitted": hasattr(self.scaler, 'mean_'),
            "label_encoder_fitted": hasattr(self.label_encoder, 'classes_')
        }
