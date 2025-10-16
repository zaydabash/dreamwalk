"""
EEG Signal Processing Module

Handles real-time EEG signal preprocessing, artifact removal, and feature extraction.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
import mne
from mne.preprocessing import ICA
from mne.filter import filter_data
from mne.channels import make_standard_montage

from ..models.signal_models import SignalData, ProcessedFeatures, EEGConfig, ProcessingConfig


class EEGProcessor:
    """EEG signal processor with artifact removal and feature extraction"""
    
    def __init__(self, config: Optional[EEGConfig] = None):
        self.config = config or EEGConfig()
        self.ica_fitted = False
        self.ica_components = None
        self.montage = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize standard montage
        try:
            self.montage = make_standard_montage('standard_1020')
        except Exception as e:
            self.logger.warning(f"Could not load standard montage: {e}")
    
    async def process(self, raw_data: SignalData) -> SignalData:
        """Process raw EEG data with filtering and artifact removal"""
        try:
            data = raw_data.data.copy()
            sampling_rate = raw_data.sampling_rate
            
            # Apply bandpass filter
            filtered_data = self._apply_bandpass_filter(data, sampling_rate)
            
            # Apply notch filter for power line noise
            if self.config.notch_filter > 0:
                filtered_data = self._apply_notch_filter(filtered_data, sampling_rate)
            
            # Apply reference
            referenced_data = self._apply_reference(filtered_data)
            
            # Remove artifacts
            cleaned_data = await self._remove_artifacts(referenced_data, sampling_rate)
            
            # Create processed signal data
            processed_data = SignalData(
                data=cleaned_data,
                sampling_rate=sampling_rate,
                channels=raw_data.channels,
                timestamp=raw_data.timestamp,
                metadata={
                    **raw_data.metadata,
                    "processing": {
                        "bandpass_applied": True,
                        "notch_filter": self.config.notch_filter,
                        "reference": self.config.reference,
                        "ica_applied": self.ica_fitted
                    }
                }
            )
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing EEG data: {e}")
            # Return original data if processing fails
            return raw_data
    
    async def process_batch(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Process batch of EEG data"""
        try:
            # Convert DataFrame to numpy array
            data = raw_data.values.T  # Transpose to get channels x samples
            
            # Create SignalData object
            signal_data = SignalData(
                data=data,
                sampling_rate=self.config.sampling_rate,
                channels=self.config.channels[:data.shape[0]],
                metadata={"batch_processing": True}
            )
            
            # Process the signal
            processed = await self.process(signal_data)
            
            # Convert back to DataFrame
            processed_df = pd.DataFrame(
                processed.data.T,
                columns=processed.channels,
                index=raw_data.index
            )
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            return raw_data
    
    def _apply_bandpass_filter(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Apply bandpass filter to EEG data"""
        try:
            filtered = filter_data(
                data,
                sfreq=sampling_rate,
                l_freq=self.config.bandpass_low,
                h_freq=self.config.bandpass_high,
                method='fir',
                phase='zero-double',
                fir_window='hamming'
            )
            return filtered
        except Exception as e:
            self.logger.error(f"Bandpass filtering failed: {e}")
            return data
    
    def _apply_notch_filter(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Apply notch filter for power line noise"""
        try:
            filtered = mne.filter.notch_filter(
                data,
                Fs=sampling_rate,
                freqs=self.config.notch_filter,
                method='fir',
                phase='zero-double'
            )
            return filtered
        except Exception as e:
            self.logger.error(f"Notch filtering failed: {e}")
            return data
    
    def _apply_reference(self, data: np.ndarray) -> np.ndarray:
        """Apply reference to EEG data"""
        try:
            if self.config.reference == "average":
                # Average reference
                avg_ref = np.mean(data, axis=0, keepdims=True)
                referenced = data - avg_ref
            elif self.config.reference == "mastoid":
                # Mastoid reference (if available)
                # For now, use average reference as fallback
                avg_ref = np.mean(data, axis=0, keepdims=True)
                referenced = data - avg_ref
            else:
                referenced = data
            
            return referenced
        except Exception as e:
            self.logger.error(f"Reference application failed: {e}")
            return data
    
    async def _remove_artifacts(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Remove artifacts using ICA"""
        try:
            if not self.ica_fitted:
                await self._fit_ica(data, sampling_rate)
            
            if self.ica_fitted and self.ica_components is not None:
                # Apply ICA artifact removal
                cleaned = self.ica_components.apply(data)
                return cleaned
            else:
                return data
                
        except Exception as e:
            self.logger.error(f"ICA artifact removal failed: {e}")
            return data
    
    async def _fit_ica(self, data: np.ndarray, sampling_rate: float):
        """Fit ICA for artifact removal"""
        try:
            # Create MNE Raw object for ICA
            info = mne.create_info(
                ch_names=self.config.channels[:data.shape[0]],
                sfreq=sampling_rate,
                ch_types='eeg'
            )
            
            raw = mne.io.RawArray(data, info)
            
            # Set montage if available
            if self.montage:
                raw.set_montage(self.montage, on_missing='warn')
            
            # Fit ICA
            n_components = min(data.shape[0], 15)  # Limit ICA components
            ica = ICA(n_components=n_components, random_state=42)
            ica.fit(raw)
            
            # Detect and exclude eye blinks and muscle artifacts
            ica.exclude = []
            
            # Find eye blink components (typically frontal channels)
            eog_indices = mne.pick_types(raw.info, eeg=False, eog=True)
            if len(eog_indices) == 0:
                # Use frontal channels as EOG proxies
                frontal_channels = [i for i, ch in enumerate(raw.ch_names) 
                                  if ch.startswith(('Fp', 'F3', 'F4', 'Fz'))]
                if frontal_channels:
                    eog_indices = frontal_channels[:1]  # Use first frontal channel
            
            if len(eog_indices) > 0:
                # Find eye blink components
                eog_scores = ica.score_sources(raw, target=eog_indices[0])
                if len(eog_scores) > 0:
                    # Exclude components with high EOG correlation
                    exclude_threshold = 0.5
                    bad_components = np.where(np.abs(eog_scores) > exclude_threshold)[0]
                    ica.exclude.extend(bad_components.tolist())
            
            # Find muscle artifact components (high frequency)
            muscle_scores = ica.score_sources(raw, target='muscle')
            if len(muscle_scores) > 0:
                exclude_threshold = 0.3
                bad_components = np.where(np.abs(muscle_scores) > exclude_threshold)[0]
                ica.exclude.extend(bad_components.tolist())
            
            self.ica_components = ica
            self.ica_fitted = True
            
            self.logger.info(f"ICA fitted with {n_components} components, "
                           f"excluding {len(ica.exclude)} components")
            
        except Exception as e:
            self.logger.error(f"ICA fitting failed: {e}")
            self.ica_fitted = False
    
    def detect_artifacts(self, data: np.ndarray, sampling_rate: float) -> Dict[str, Any]:
        """Detect artifacts in EEG data"""
        artifacts = {
            "eye_blinks": 0,
            "muscle_artifacts": 0,
            "flat_channels": 0,
            "bad_channels": []
        }
        
        try:
            # Detect eye blinks (high amplitude in frontal channels)
            frontal_channels = [i for i, ch in enumerate(self.config.channels) 
                              if ch.startswith(('Fp', 'F3', 'F4', 'Fz'))]
            
            if frontal_channels:
                frontal_data = data[frontal_channels]
                blink_threshold = 3 * np.std(frontal_data)
                blink_epochs = np.abs(frontal_data) > blink_threshold
                artifacts["eye_blinks"] = int(np.sum(blink_epochs) / len(frontal_channels))
            
            # Detect muscle artifacts (high frequency activity)
            high_freq = filter_data(
                data,
                sfreq=sampling_rate,
                l_freq=20,
                h_freq=45,
                method='fir'
            )
            
            muscle_threshold = 2 * np.std(high_freq)
            muscle_epochs = np.abs(high_freq) > muscle_threshold
            artifacts["muscle_artifacts"] = int(np.sum(muscle_epochs) / data.shape[0])
            
            # Detect flat channels
            for i, channel_data in enumerate(data):
                if np.std(channel_data) < 1e-8:
                    artifacts["flat_channels"] += 1
                    artifacts["bad_channels"].append(i)
            
        except Exception as e:
            self.logger.error(f"Artifact detection failed: {e}")
        
        return artifacts
    
    def get_signal_quality_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate signal quality metrics"""
        metrics = {}
        
        try:
            # Signal-to-noise ratio approximation
            signal_power = np.var(data)
            noise_power = np.var(np.diff(data, axis=1))
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            metrics["snr_db"] = float(snr)
            
            # Channel correlation (should be low for clean EEG)
            channel_corr = np.corrcoef(data)
            np.fill_diagonal(channel_corr, 0)
            metrics["channel_correlation"] = float(np.mean(np.abs(channel_corr)))
            
            # Amplitude range
            metrics["amplitude_range"] = float(np.max(data) - np.min(data))
            
            # Frequency content (check for proper EEG bands)
            freqs, psd = signal.welch(data, fs=self.config.sampling_rate, nperseg=1024)
            
            # Alpha peak detection (should be prominent in relaxed states)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            alpha_power = np.mean(psd[:, alpha_mask])
            total_power = np.mean(psd)
            metrics["alpha_ratio"] = float(alpha_power / total_power)
            
        except Exception as e:
            self.logger.error(f"Quality metrics calculation failed: {e}")
            metrics = {"error": str(e)}
        
        return metrics
