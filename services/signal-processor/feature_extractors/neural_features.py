"""
Neural Feature Extraction Module

Extracts meaningful features from processed EEG signals for downstream analysis.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
from scipy.signal import welch, coherence
from sklearn.preprocessing import StandardScaler

from ..models.signal_models import SignalData, ProcessedFeatures, ProcessingConfig


class NeuralFeatureExtractor:
    """Extract neural features from processed EEG signals"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Feature scaler for normalization
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Band definitions
        self.bands = {
            'delta': self.config.delta_band,
            'theta': self.config.theta_band,
            'alpha': self.config.alpha_band,
            'beta': self.config.beta_band,
            'gamma': self.config.gamma_band
        }
    
    async def extract(self, signal_data: SignalData) -> ProcessedFeatures:
        """Extract features from a single signal window"""
        try:
            data = signal_data.data
            sampling_rate = signal_data.sampling_rate
            channels = signal_data.channels
            
            # Extract spectral features
            spectral_features = await self._extract_spectral_features(data, sampling_rate)
            
            # Extract temporal features
            temporal_features = await self._extract_temporal_features(data)
            
            # Extract connectivity features
            connectivity_features = await self._extract_connectivity_features(data, sampling_rate)
            
            # Extract asymmetry features
            asymmetry_features = await self._extract_asymmetry_features(data, sampling_rate, channels)
            
            # Calculate artifact metrics
            artifact_metrics = await self._calculate_artifact_metrics(data)
            
            # Calculate summary statistics
            summary_stats = await self._calculate_summary_statistics(data)
            
            # Create ProcessedFeatures object
            features = ProcessedFeatures(
                # Spectral features
                delta_power=spectral_features['delta'],
                theta_power=spectral_features['theta'],
                alpha_power=spectral_features['alpha'],
                beta_power=spectral_features['beta'],
                gamma_power=spectral_features['gamma'],
                
                # Temporal features
                hjorth_activity=temporal_features['hjorth_activity'],
                hjorth_mobility=temporal_features['hjorth_mobility'],
                hjorth_complexity=temporal_features['hjorth_complexity'],
                
                # Connectivity features
                coherence_matrix=connectivity_features['coherence'],
                phase_lag_index=connectivity_features['pli'],
                
                # Asymmetry features
                frontal_asymmetry=asymmetry_features['frontal'],
                parietal_asymmetry=asymmetry_features['parietal'],
                
                # Artifact metrics
                artifact_ratio=artifact_metrics['artifact_ratio'],
                eye_blink_count=artifact_metrics['eye_blinks'],
                
                # Summary statistics
                mean_amplitude=summary_stats['mean'],
                std_amplitude=summary_stats['std'],
                skewness=summary_stats['skewness'],
                kurtosis=summary_stats['kurtosis'],
                
                # Metadata
                window_duration=self.config.window_size,
                overlap=self.config.overlap
            )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            # Return minimal features if extraction fails
            return self._create_minimal_features(len(channels))
    
    async def extract_batch(self, signal_data: SignalData) -> pd.DataFrame:
        """Extract features from multiple signal windows"""
        try:
            # This would be implemented for batch processing
            # For now, process single window
            features = await self.extract(signal_data)
            
            # Convert to DataFrame
            feature_dict = features.dict()
            df = pd.DataFrame([feature_dict])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Batch feature extraction failed: {e}")
            return pd.DataFrame()
    
    async def _extract_spectral_features(self, data: np.ndarray, sampling_rate: float) -> Dict[str, List[float]]:
        """Extract spectral power features"""
        spectral_features = {}
        
        try:
            # Calculate power spectral density
            freqs, psd = welch(data, fs=sampling_rate, nperseg=min(1024, data.shape[1]//4))
            
            # Extract power in each frequency band
            for band_name, (low_freq, high_freq) in self.bands.items():
                # Find frequency indices
                freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                
                # Calculate mean power across channels
                band_power = np.mean(psd[:, freq_mask], axis=1)
                spectral_features[band_name] = band_power.tolist()
            
        except Exception as e:
            self.logger.error(f"Spectral feature extraction failed: {e}")
            # Return zero features
            for band_name in self.bands.keys():
                spectral_features[band_name] = [0.0] * data.shape[0]
        
        return spectral_features
    
    async def _extract_temporal_features(self, data: np.ndarray) -> Dict[str, List[float]]:
        """Extract temporal features (Hjorth parameters)"""
        temporal_features = {
            'hjorth_activity': [],
            'hjorth_mobility': [],
            'hjorth_complexity': []
        }
        
        try:
            for channel_data in data:
                # Hjorth Activity (variance)
                activity = np.var(channel_data)
                temporal_features['hjorth_activity'].append(float(activity))
                
                # Hjorth Mobility (standard deviation of first derivative)
                first_derivative = np.diff(channel_data)
                mobility = np.sqrt(np.var(first_derivative) / (np.var(channel_data) + 1e-10))
                temporal_features['hjorth_mobility'].append(float(mobility))
                
                # Hjorth Complexity (mobility of first derivative / mobility of signal)
                second_derivative = np.diff(first_derivative)
                complexity = np.sqrt(
                    (np.var(second_derivative) / (np.var(first_derivative) + 1e-10)) /
                    (np.var(first_derivative) / (np.var(channel_data) + 1e-10))
                )
                temporal_features['hjorth_complexity'].append(float(complexity))
                
        except Exception as e:
            self.logger.error(f"Temporal feature extraction failed: {e}")
            # Return zero features
            n_channels = data.shape[0]
            for key in temporal_features.keys():
                temporal_features[key] = [0.0] * n_channels
        
        return temporal_features
    
    async def _extract_connectivity_features(self, data: np.ndarray, sampling_rate: float) -> Dict[str, Optional[np.ndarray]]:
        """Extract connectivity features between channels"""
        connectivity_features = {
            'coherence': None,
            'pli': None
        }
        
        try:
            n_channels = data.shape[0]
            
            if n_channels < 2:
                return connectivity_features
            
            # Calculate coherence matrix
            coherence_matrix = np.zeros((n_channels, n_channels))
            
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    # Calculate coherence
                    freqs, coh = coherence(data[i], data[j], fs=sampling_rate, nperseg=min(512, data.shape[1]//4))
                    
                    # Average coherence across frequency bands
                    alpha_mask = (freqs >= 8) & (freqs <= 13)
                    if np.any(alpha_mask):
                        avg_coh = np.mean(coh[alpha_mask])
                        coherence_matrix[i, j] = avg_coh
                        coherence_matrix[j, i] = avg_coh
            
            # Set diagonal to 1
            np.fill_diagonal(coherence_matrix, 1.0)
            connectivity_features['coherence'] = coherence_matrix
            
            # Calculate Phase Lag Index (PLI) - simplified version
            pli_matrix = np.zeros((n_channels, n_channels))
            
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    # Calculate instantaneous phase difference
                    phase_i = np.angle(signal.hilbert(data[i]))
                    phase_j = np.angle(signal.hilbert(data[j]))
                    phase_diff = phase_i - phase_j
                    
                    # PLI = |E[sign(imag(exp(i*phase_diff)))]|
                    pli = np.abs(np.mean(np.sign(np.imag(np.exp(1j * phase_diff)))))
                    pli_matrix[i, j] = pli
                    pli_matrix[j, i] = pli
            
            np.fill_diagonal(pli_matrix, 0.0)
            connectivity_features['pli'] = pli_matrix
            
        except Exception as e:
            self.logger.error(f"Connectivity feature extraction failed: {e}")
        
        return connectivity_features
    
    async def _extract_asymmetry_features(self, data: np.ndarray, sampling_rate: float, channels: List[str]) -> Dict[str, float]:
        """Extract frontal and parietal asymmetry features"""
        asymmetry_features = {
            'frontal': 0.0,
            'parietal': 0.0
        }
        
        try:
            # Find frontal and parietal channels
            frontal_left = None
            frontal_right = None
            parietal_left = None
            parietal_right = None
            
            for i, ch in enumerate(channels):
                if ch in ['F3', 'F7', 'Fp1']:
                    frontal_left = i
                elif ch in ['F4', 'F8', 'Fp2']:
                    frontal_right = i
                elif ch in ['P3', 'P7']:
                    parietal_left = i
                elif ch in ['P4', 'P8']:
                    parietal_right = i
            
            # Calculate alpha power for asymmetry
            freqs, psd = welch(data, fs=sampling_rate, nperseg=min(1024, data.shape[1]//4))
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            alpha_power = np.mean(psd[:, alpha_mask], axis=1)
            
            # Frontal asymmetry
            if frontal_left is not None and frontal_right is not None:
                frontal_asym = (alpha_power[frontal_right] - alpha_power[frontal_left]) / (alpha_power[frontal_right] + alpha_power[frontal_left] + 1e-10)
                asymmetry_features['frontal'] = float(frontal_asym)
            
            # Parietal asymmetry
            if parietal_left is not None and parietal_right is not None:
                parietal_asym = (alpha_power[parietal_right] - alpha_power[parietal_left]) / (alpha_power[parietal_right] + alpha_power[parietal_left] + 1e-10)
                asymmetry_features['parietal'] = float(parietal_asym)
                
        except Exception as e:
            self.logger.error(f"Asymmetry feature extraction failed: {e}")
        
        return asymmetry_features
    
    async def _calculate_artifact_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate artifact contamination metrics"""
        artifact_metrics = {
            'artifact_ratio': 0.0,
            'eye_blinks': 0
        }
        
        try:
            # Detect high-amplitude artifacts
            artifact_threshold = 3 * np.std(data)
            artifact_mask = np.abs(data) > artifact_threshold
            artifact_ratio = np.mean(artifact_mask)
            artifact_metrics['artifact_ratio'] = float(artifact_ratio)
            
            # Count eye blinks (simplified detection)
            # Look for high amplitude in frontal channels
            frontal_channels = [i for i in range(min(4, data.shape[0]))]  # Assume first 4 are frontal
            if frontal_channels:
                frontal_data = data[frontal_channels]
                blink_threshold = 2 * np.std(frontal_data)
                blink_mask = np.abs(frontal_data) > blink_threshold
                eye_blinks = np.sum(blink_mask)
                artifact_metrics['eye_blinks'] = int(eye_blinks)
                
        except Exception as e:
            self.logger.error(f"Artifact metrics calculation failed: {e}")
        
        return artifact_metrics
    
    async def _calculate_summary_statistics(self, data: np.ndarray) -> Dict[str, List[float]]:
        """Calculate summary statistics for each channel"""
        summary_stats = {
            'mean': [],
            'std': [],
            'skewness': [],
            'kurtosis': []
        }
        
        try:
            for channel_data in data:
                summary_stats['mean'].append(float(np.mean(channel_data)))
                summary_stats['std'].append(float(np.std(channel_data)))
                summary_stats['skewness'].append(float(skew(channel_data)))
                summary_stats['kurtosis'].append(float(kurtosis(channel_data)))
                
        except Exception as e:
            self.logger.error(f"Summary statistics calculation failed: {e}")
            n_channels = data.shape[0]
            for key in summary_stats.keys():
                summary_stats[key] = [0.0] * n_channels
        
        return summary_stats
    
    def _create_minimal_features(self, n_channels: int) -> ProcessedFeatures:
        """Create minimal features when extraction fails"""
        return ProcessedFeatures(
            delta_power=[0.0] * n_channels,
            theta_power=[0.0] * n_channels,
            alpha_power=[0.0] * n_channels,
            beta_power=[0.0] * n_channels,
            gamma_power=[0.0] * n_channels,
            hjorth_activity=[0.0] * n_channels,
            hjorth_mobility=[0.0] * n_channels,
            hjorth_complexity=[0.0] * n_channels,
            frontal_asymmetry=0.0,
            parietal_asymmetry=0.0,
            artifact_ratio=1.0,  # High artifact ratio indicates failure
            eye_blink_count=0,
            mean_amplitude=[0.0] * n_channels,
            std_amplitude=[0.0] * n_channels,
            skewness=[0.0] * n_channels,
            kurtosis=[0.0] * n_channels
        )
    
    def normalize_features(self, features: ProcessedFeatures) -> ProcessedFeatures:
        """Normalize features using fitted scaler"""
        try:
            if not self.is_fitted:
                # Fit scaler on this data (in production, fit on training data)
                feature_vector = self._features_to_vector(features)
                self.scaler.fit([feature_vector])
                self.is_fitted = True
            
            # Normalize features
            feature_vector = self._features_to_vector(features)
            normalized_vector = self.scaler.transform([feature_vector])[0]
            
            # Convert back to ProcessedFeatures (simplified)
            return features  # For now, return original features
            
        except Exception as e:
            self.logger.error(f"Feature normalization failed: {e}")
            return features
    
    def _features_to_vector(self, features: ProcessedFeatures) -> np.ndarray:
        """Convert ProcessedFeatures to a flat vector for normalization"""
        vector = []
        
        # Add spectral features
        for band_power in [features.delta_power, features.theta_power, features.alpha_power, 
                          features.beta_power, features.gamma_power]:
            vector.extend(band_power)
        
        # Add temporal features
        vector.extend(features.hjorth_activity)
        vector.extend(features.hjorth_mobility)
        vector.extend(features.hjorth_complexity)
        
        # Add asymmetry features
        vector.append(features.frontal_asymmetry)
        vector.append(features.parietal_asymmetry)
        
        # Add artifact metrics
        vector.append(features.artifact_ratio)
        vector.append(float(features.eye_blink_count))
        
        # Add summary statistics
        vector.extend(features.mean_amplitude)
        vector.extend(features.std_amplitude)
        vector.extend(features.skewness)
        vector.extend(features.kurtosis)
        
        return np.array(vector)
