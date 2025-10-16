"""
Mock EEG Streamer

Generates realistic synthetic EEG data for testing and demonstration purposes.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import AsyncGenerator, Dict, Any, List, Optional
from datetime import datetime
import random

from ..models.signal_models import SignalData, EEGConfig


class MockStreamer:
    """Generate realistic synthetic EEG data streams"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default EEG configuration
        self.eeg_config = EEGConfig(
            sampling_rate=self.config.get('sampling_rate', 250),
            channels=self.config.get('channels', [
                "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"
            ])
        )
        
        # Stream parameters
        self.window_size = self.config.get('window_size', 4.0)  # seconds
        self.update_rate = self.config.get('update_rate', 10)  # Hz
        
        # State variables
        self.current_state = "neutral"
        self.state_transition_prob = 0.1
        self.samples_generated = 0
        
        # Emotional states and their characteristics
        self.emotional_states = {
            "relaxed": {
                "alpha_power": 1.2,
                "theta_power": 0.8,
                "beta_power": 0.3,
                "gamma_power": 0.2,
                "asymmetry": 0.1,
                "noise_level": 0.1
            },
            "focused": {
                "alpha_power": 0.6,
                "theta_power": 0.4,
                "beta_power": 1.5,
                "gamma_power": 1.0,
                "asymmetry": -0.2,
                "noise_level": 0.15
            },
            "stressed": {
                "alpha_power": 0.4,
                "theta_power": 0.3,
                "beta_power": 2.0,
                "gamma_power": 1.5,
                "asymmetry": -0.4,
                "noise_level": 0.25
            },
            "meditative": {
                "alpha_power": 1.8,
                "theta_power": 1.5,
                "beta_power": 0.2,
                "gamma_power": 0.1,
                "asymmetry": 0.3,
                "noise_level": 0.05
            },
            "neutral": {
                "alpha_power": 1.0,
                "theta_power": 1.0,
                "beta_power": 1.0,
                "gamma_power": 1.0,
                "asymmetry": 0.0,
                "noise_level": 0.2
            }
        }
        
        # Initialize signal parameters
        self._initialize_signal_parameters()
    
    def _initialize_signal_parameters(self):
        """Initialize signal generation parameters"""
        self.frequencies = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        # Channel-specific parameters
        self.channel_parameters = {}
        for i, channel in enumerate(self.eeg_config.channels):
            self.channel_parameters[channel] = {
                'base_amplitude': random.uniform(10, 50),
                'phase_offset': random.uniform(0, 2 * np.pi),
                'noise_factor': random.uniform(0.5, 1.5)
            }
    
    async def stream(self) -> AsyncGenerator[SignalData, None]:
        """Generate continuous stream of synthetic EEG data"""
        try:
            while True:
                # Generate data window
                signal_data = await self._generate_signal_window()
                
                yield signal_data
                
                # Wait for next update
                await asyncio.sleep(1.0 / self.update_rate)
                
        except Exception as e:
            self.logger.error(f"Error in mock stream: {e}")
    
    async def _generate_signal_window(self) -> SignalData:
        """Generate a single window of synthetic EEG data"""
        try:
            # Update emotional state occasionally
            await self._update_emotional_state()
            
            # Get current state parameters
            state_params = self.emotional_states[self.current_state]
            
            # Calculate number of samples
            n_samples = int(self.window_size * self.eeg_config.sampling_rate)
            n_channels = len(self.eeg_config.channels)
            
            # Initialize data array
            data = np.zeros((n_channels, n_samples))
            
            # Generate signal for each channel
            for ch_idx, channel in enumerate(self.eeg_config.channels):
                channel_data = await self._generate_channel_signal(
                    n_samples, channel, state_params, ch_idx
                )
                data[ch_idx, :] = channel_data
            
            # Add some inter-channel correlation
            data = await self._add_channel_correlation(data)
            
            # Add artifacts occasionally
            if random.random() < 0.1:  # 10% chance of artifacts
                data = await self._add_artifacts(data)
            
            # Create SignalData object
            signal_data = SignalData(
                data=data,
                sampling_rate=self.eeg_config.sampling_rate,
                channels=self.eeg_config.channels,
                metadata={
                    "source": "mock_streamer",
                    "emotional_state": self.current_state,
                    "state_parameters": state_params,
                    "window_index": self.samples_generated,
                    "has_artifacts": random.random() < 0.1
                }
            )
            
            self.samples_generated += 1
            return signal_data
            
        except Exception as e:
            self.logger.error(f"Error generating signal window: {e}")
            # Return minimal data on error
            return self._create_minimal_signal()
    
    async def _generate_channel_signal(self, n_samples: int, channel: str, 
                                     state_params: Dict[str, float], ch_idx: int) -> np.ndarray:
        """Generate signal for a single channel"""
        try:
            # Get channel parameters
            ch_params = self.channel_parameters[channel]
            
            # Time vector
            t = np.arange(n_samples) / self.eeg_config.sampling_rate
            
            # Initialize signal
            signal = np.zeros(n_samples)
            
            # Add frequency components
            for band, (low_freq, high_freq) in self.frequencies.items():
                # Get power multiplier for this band
                power_multiplier = state_params.get(f"{band}_power", 1.0)
                
                # Generate multiple frequency components within band
                n_components = random.randint(2, 5)
                for _ in range(n_components):
                    freq = random.uniform(low_freq, high_freq)
                    amplitude = ch_params['base_amplitude'] * power_multiplier * random.uniform(0.1, 0.5)
                    phase = ch_params['phase_offset'] + random.uniform(0, 2 * np.pi)
                    
                    # Add sinusoidal component
                    component = amplitude * np.sin(2 * np.pi * freq * t + phase)
                    signal += component
            
            # Add asymmetry effect (affects frontal channels)
            if channel.startswith(('F', 'P')):  # Frontal or parietal channels
                asymmetry = state_params.get('asymmetry', 0.0)
                if 'F' in channel and '3' in channel:  # Left frontal
                    signal *= (1 - asymmetry * 0.3)
                elif 'F' in channel and '4' in channel:  # Right frontal
                    signal *= (1 + asymmetry * 0.3)
                elif 'P' in channel and '3' in channel:  # Left parietal
                    signal *= (1 - asymmetry * 0.2)
                elif 'P' in channel and '4' in channel:  # Right parietal
                    signal *= (1 + asymmetry * 0.2)
            
            # Add noise
            noise_level = state_params.get('noise_level', 0.2)
            noise = np.random.normal(0, noise_level * ch_params['noise_factor'], n_samples)
            signal += noise
            
            # Add slow drift
            drift = np.cumsum(np.random.normal(0, 0.01, n_samples))
            signal += drift
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating channel signal: {e}")
            return np.random.normal(0, 1, n_samples)
    
    async def _add_channel_correlation(self, data: np.ndarray) -> np.ndarray:
        """Add realistic inter-channel correlation"""
        try:
            # Create correlation matrix
            n_channels = data.shape[0]
            correlation_strength = 0.1  # Weak correlation
            
            # Apply correlation between nearby channels
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    if abs(i - j) <= 2:  # Only correlate nearby channels
                        correlation = correlation_strength * np.random.normal(0, 1)
                        data[j] += correlation * data[i]
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error adding channel correlation: {e}")
            return data
    
    async def _add_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Add realistic artifacts to the signal"""
        try:
            n_channels, n_samples = data.shape
            
            # Eye blink artifacts (affect frontal channels)
            if random.random() < 0.3:  # 30% chance of eye blink
                blink_start = random.randint(0, n_samples - 100)
                blink_duration = random.randint(20, 80)
                blink_end = min(blink_start + blink_duration, n_samples)
                
                # Create eye blink artifact
                blink_artifact = np.zeros(blink_end - blink_start)
                blink_artifact = np.exp(-0.1 * np.arange(len(blink_artifact)))
                blink_artifact *= random.uniform(50, 150)
                
                # Apply to frontal channels
                frontal_channels = [i for i, ch in enumerate(self.eeg_config.channels) 
                                  if ch.startswith('F')]
                for ch_idx in frontal_channels:
                    data[ch_idx, blink_start:blink_end] += blink_artifact
            
            # Muscle artifacts (affect all channels)
            if random.random() < 0.2:  # 20% chance of muscle artifact
                muscle_start = random.randint(0, n_samples - 200)
                muscle_duration = random.randint(50, 150)
                muscle_end = min(muscle_start + muscle_duration, n_samples)
                
                # Create high-frequency muscle artifact
                t = np.arange(muscle_end - muscle_start) / self.eeg_config.sampling_rate
                muscle_freq = random.uniform(20, 80)
                muscle_artifact = random.uniform(20, 80) * np.sin(2 * np.pi * muscle_freq * t)
                muscle_artifact *= np.exp(-0.05 * np.arange(len(muscle_artifact)))
                
                # Apply to all channels with varying intensity
                for ch_idx in range(n_channels):
                    intensity = random.uniform(0.5, 1.5)
                    data[ch_idx, muscle_start:muscle_end] += intensity * muscle_artifact
            
            # Electrode pop artifacts
            if random.random() < 0.1:  # 10% chance of electrode pop
                pop_channel = random.randint(0, n_channels - 1)
                pop_start = random.randint(0, n_samples - 10)
                pop_duration = random.randint(5, 15)
                pop_end = min(pop_start + pop_duration, n_samples)
                
                # Create electrode pop
                pop_artifact = np.random.normal(0, 100, pop_end - pop_start)
                data[pop_channel, pop_start:pop_end] += pop_artifact
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error adding artifacts: {e}")
            return data
    
    async def _update_emotional_state(self):
        """Update emotional state with smooth transitions"""
        try:
            # Random state transitions
            if random.random() < self.state_transition_prob:
                available_states = [s for s in self.emotional_states.keys() if s != self.current_state]
                if available_states:
                    new_state = random.choice(available_states)
                    self.logger.info(f"Emotional state transition: {self.current_state} -> {new_state}")
                    self.current_state = new_state
            
            # Adjust transition probability based on current state
            if self.current_state == "neutral":
                self.state_transition_prob = 0.1
            else:
                self.state_transition_prob = 0.05  # Lower probability to stay in emotional states
            
        except Exception as e:
            self.logger.error(f"Error updating emotional state: {e}")
    
    def _create_minimal_signal(self) -> SignalData:
        """Create minimal signal data when generation fails"""
        n_samples = int(self.window_size * self.eeg_config.sampling_rate)
        n_channels = len(self.eeg_config.channels)
        
        # Generate simple noise
        data = np.random.normal(0, 1, (n_channels, n_samples))
        
        return SignalData(
            data=data,
            sampling_rate=self.eeg_config.sampling_rate,
            channels=self.eeg_config.channels,
            metadata={
                "source": "mock_streamer",
                "error": "minimal_signal_generated",
                "emotional_state": "neutral"
            }
        )
    
    def set_emotional_state(self, state: str):
        """Manually set emotional state"""
        if state in self.emotional_states:
            self.current_state = state
            self.logger.info(f"Emotional state manually set to: {state}")
        else:
            self.logger.warning(f"Unknown emotional state: {state}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current emotional state and parameters"""
        return {
            "state": self.current_state,
            "parameters": self.emotional_states[self.current_state],
            "samples_generated": self.samples_generated
        }
