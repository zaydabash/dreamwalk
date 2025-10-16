"""
EEG to CLIP Embedding Decoder

Maps neural features to CLIP embedding space for world generation.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import clip

from ..models.decoder_models import DecoderConfig, TrainingData


class EEGToCLIPDataset(Dataset):
    """Dataset for EEG to CLIP mapping"""
    
    def __init__(self, features: List[Dict[str, Any]], targets: List[np.ndarray], 
                 scaler: Optional[StandardScaler] = None):
        self.features = features
        self.targets = targets
        self.scaler = scaler
        
        # Convert features to tensor format
        self.feature_vectors = self._extract_feature_vectors()
        
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
        target = torch.FloatTensor(self.targets[idx])
        return feature_vector, target


class EEGToCLIPNet(nn.Module):
    """Neural network for mapping EEG features to CLIP embeddings"""
    
    def __init__(self, input_dim: int, output_dim: int = 512, hidden_dims: List[int] = None):
        super(EEGToCLIPNet, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512, 256]
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
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
        return self.network(x)


class EEGToCLIPDecoder:
    """EEG to CLIP embedding decoder"""
    
    def __init__(self, clip_model=None, model_path: Optional[str] = None, 
                 config: Optional[DecoderConfig] = None):
        self.clip_model = clip_model
        self.model_path = model_path
        self.config = config or DecoderConfig()
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.device = self._get_device()
        
        # Model parameters
        self.input_dim = 88  # Calculated from feature extraction
        self.output_dim = 512  # CLIP embedding dimension
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
                self.model = EEGToCLIPNet(
                    input_dim=checkpoint.get('input_dim', self.input_dim),
                    output_dim=checkpoint.get('output_dim', self.output_dim),
                    hidden_dims=checkpoint.get('hidden_dims', [256, 512, 256])
                )
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                # Load scaler if available
                if 'scaler' in checkpoint:
                    self.scaler = checkpoint['scaler']
                
                self.is_trained = True
                self.logger.info("Model loaded successfully", model_path=path)
                
            else:
                self.logger.warning("Model file not found, will train from scratch", path=path)
                
        except Exception as e:
            self.logger.error("Failed to load model", error=str(e))
            raise
    
    async def save_model(self, model_path: str):
        """Save trained model"""
        try:
            if self.model is None:
                raise ValueError("No model to save")
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'hidden_dims': [256, 512, 256],
                'scaler': self.scaler,
                'config': self.config.dict()
            }
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(checkpoint, model_path)
            
            self.logger.info("Model saved successfully", model_path=model_path)
            
        except Exception as e:
            self.logger.error("Failed to save model", error=str(e))
            raise
    
    async def decode(self, features: Dict[str, Any]) -> np.ndarray:
        """Decode neural features to CLIP embedding"""
        try:
            if not self.is_trained or self.model is None:
                # Return random embedding if model not available
                self.logger.warning("Model not trained, returning random embedding")
                return np.random.normal(0, 1, self.output_dim)
            
            # Extract feature vector
            feature_vector = self._extract_feature_vector(features)
            
            # Normalize if scaler is fitted
            if hasattr(self.scaler, 'mean_'):
                feature_vector = self.scaler.transform([feature_vector])[0]
            
            # Convert to tensor
            feature_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                embedding = self.model(feature_tensor)
                embedding = embedding.cpu().numpy()[0]
            
            # Normalize embedding to unit sphere (like CLIP)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            self.logger.error("Decoding failed", error=str(e))
            # Return neutral embedding on error
            return np.zeros(self.output_dim)
    
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
            self.logger.info("Starting synthetic training", n_samples=len(training_data.features))
            
            # Prepare data
            features = training_data.features
            targets = training_data.targets
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            # Create datasets
            train_dataset = EEGToCLIPDataset(X_train, y_train, scaler=None)
            val_dataset = EEGToCLIPDataset(X_val, y_val, scaler=train_dataset.scaler)
            
            # Fit scaler on training data
            self.scaler.fit(train_dataset.feature_vectors)
            
            # Recreate datasets with fitted scaler
            train_dataset = EEGToCLIPDataset(X_train, y_train, scaler=self.scaler)
            val_dataset = EEGToCLIPDataset(X_val, y_val, scaler=self.scaler)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Initialize model
            self.model = EEGToCLIPNet(
                input_dim=train_dataset.feature_vectors.shape[1],
                output_dim=self.output_dim
            ).to(self.device)
            
            # Training setup
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            criterion = nn.CosineEmbeddingLoss()
            
            # Training loop
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(100):  # Maximum epochs
                # Training
                train_loss = await self._train_epoch(train_loader, optimizer, criterion)
                
                # Validation
                val_loss = await self._validate_epoch(val_loader, criterion)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                self.logger.info(
                    "Epoch completed",
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
            self.logger.info("Training completed", best_val_loss=best_val_loss)
            
        except Exception as e:
            self.logger.error("Synthetic training failed", error=str(e))
            raise
    
    async def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                          criterion: nn.Module) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch_features)
            
            # Compute loss (cosine similarity loss)
            target_ones = torch.ones(batch_features.size(0)).to(self.device)
            loss = criterion(predictions, batch_targets, target_ones)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    async def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                predictions = self.model(batch_features)
                
                # Compute loss
                target_ones = torch.ones(batch_features.size(0)).to(self.device)
                loss = criterion(predictions, batch_targets, target_ones)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "is_trained": self.is_trained,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "device": str(self.device),
            "model_path": self.model_path,
            "scaler_fitted": hasattr(self.scaler, 'mean_')
        }
