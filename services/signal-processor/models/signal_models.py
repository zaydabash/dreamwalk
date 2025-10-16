"""
Signal data models for DreamWalk
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
import numpy as np


class EEGConfig(BaseModel):
    """EEG signal configuration"""
    sampling_rate: int = Field(default=250, description="Sampling rate in Hz")
    channels: List[str] = Field(default_factory=lambda: [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"
    ], description="EEG channel names")
    reference: str = Field(default="average", description="Reference type")
    notch_filter: float = Field(default=60.0, description="Notch filter frequency")
    bandpass_low: float = Field(default=1.0, description="Low-pass filter cutoff")
    bandpass_high: float = Field(default=45.0, description="High-pass filter cutoff")


class fMRIConfig(BaseModel):
    """fMRI signal configuration"""
    voxel_size: List[float] = Field(default=[3.0, 3.0, 3.0], description="Voxel size in mm")
    tr: float = Field(default=2.0, description="Repetition time in seconds")
    te: float = Field(default=30.0, description="Echo time in ms")
    slice_timing: str = Field(default="ascending", description="Slice timing order")
    spatial_smoothing: float = Field(default=6.0, description="FWHM in mm")


class SignalData(BaseModel):
    """Raw signal data container"""
    data: np.ndarray = Field(..., description="Signal data array")
    sampling_rate: float = Field(..., description="Sampling rate")
    channels: List[str] = Field(..., description="Channel names")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Data timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        arbitrary_types_allowed = True


class ProcessedFeatures(BaseModel):
    """Extracted neural features"""
    # Spectral features
    delta_power: List[float] = Field(..., description="Delta band power (1-4 Hz)")
    theta_power: List[float] = Field(..., description="Theta band power (4-8 Hz)")
    alpha_power: List[float] = Field(..., description="Alpha band power (8-13 Hz)")
    beta_power: List[float] = Field(..., description="Beta band power (13-30 Hz)")
    gamma_power: List[float] = Field(..., description="Gamma band power (30-45 Hz)")
    
    # Temporal features
    hjorth_activity: List[float] = Field(..., description="Hjorth activity parameter")
    hjorth_mobility: List[float] = Field(..., description="Hjorth mobility parameter")
    hjorth_complexity: List[float] = Field(..., description="Hjorth complexity parameter")
    
    # Connectivity features
    coherence_matrix: Optional[np.ndarray] = Field(None, description="Channel coherence matrix")
    phase_lag_index: Optional[np.ndarray] = Field(None, description="Phase lag index matrix")
    
    # Asymmetry features
    frontal_asymmetry: float = Field(..., description="Frontal alpha asymmetry")
    parietal_asymmetry: float = Field(..., description="Parietal alpha asymmetry")
    
    # Artifact metrics
    artifact_ratio: float = Field(..., description="Ratio of artifact-contaminated epochs")
    eye_blink_count: int = Field(default=0, description="Number of eye blinks detected")
    
    # Summary statistics
    mean_amplitude: List[float] = Field(..., description="Mean amplitude per channel")
    std_amplitude: List[float] = Field(..., description="Standard deviation per channel")
    skewness: List[float] = Field(..., description="Skewness per channel")
    kurtosis: List[float] = Field(..., description="Kurtosis per channel")
    
    # Metadata
    window_duration: float = Field(default=4.0, description="Analysis window duration in seconds")
    overlap: float = Field(default=0.5, description="Window overlap ratio")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Feature extraction timestamp")
    
    class Config:
        arbitrary_types_allowed = True


class EmotionalState(BaseModel):
    """Estimated emotional state"""
    valence: float = Field(..., ge=-1.0, le=1.0, description="Valence (-1 to 1)")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Arousal (0 to 1)")
    dominance: float = Field(..., ge=0.0, le=1.0, description="Dominance (0 to 1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Estimation confidence")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Estimation timestamp")


class NeuralMotif(BaseModel):
    """Detected neural motif/pattern"""
    motif_id: str = Field(..., description="Unique motif identifier")
    motif_type: str = Field(..., description="Type of motif (e.g., 'meditation', 'stress', 'focus')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    features: Dict[str, float] = Field(..., description="Feature values that define this motif")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")


class WorldState(BaseModel):
    """World generation state derived from neural data"""
    # Core emotional state
    emotional_state: EmotionalState = Field(..., description="Current emotional state")
    
    # Detected motifs
    motifs: List[NeuralMotif] = Field(default_factory=list, description="Active neural motifs")
    
    # Latent embeddings
    clip_embedding: List[float] = Field(..., description="CLIP latent embedding (768 dim)")
    semantic_embedding: List[float] = Field(..., description="Semantic embedding (512 dim)")
    
    # World parameters
    biome_type: str = Field(default="neutral", description="Primary biome type")
    weather_intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="Weather intensity")
    lighting_mood: str = Field(default="neutral", description="Lighting mood")
    color_palette: List[float] = Field(..., description="RGB color palette (normalized)")
    object_density: float = Field(default=0.5, ge=0.0, le=1.0, description="Object placement density")
    structure_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Level of structural organization")
    
    # Audio parameters
    ambient_volume: float = Field(default=0.5, ge=0.0, le=1.0, description="Ambient audio volume")
    music_intensity: float = Field(default=0.3, ge=0.0, le=1.0, description="Music intensity")
    sound_effects: List[str] = Field(default_factory=list, description="Active sound effects")
    
    # Temporal dynamics
    change_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Rate of environmental changes")
    morph_speed: float = Field(default=1.0, ge=0.1, le=5.0, description="World morphing speed multiplier")
    
    # Metadata
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="State timestamp")
    version: str = Field(default="1.0", description="World state schema version")


class ProcessingConfig(BaseModel):
    """Configuration for signal processing pipeline"""
    # General settings
    window_size: float = Field(default=4.0, description="Analysis window size in seconds")
    overlap: float = Field(default=0.75, description="Window overlap ratio")
    min_quality_threshold: float = Field(default=0.3, description="Minimum signal quality threshold")
    
    # Filtering settings
    enable_ica: bool = Field(default=True, description="Enable ICA artifact removal")
    ica_components: Optional[int] = Field(default=None, description="Number of ICA components")
    enable_artifact_detection: bool = Field(default=True, description="Enable artifact detection")
    
    # Feature extraction settings
    extract_spectral: bool = Field(default=True, description="Extract spectral features")
    extract_temporal: bool = Field(default=True, description="Extract temporal features")
    extract_connectivity: bool = Field(default=True, description="Extract connectivity features")
    extract_asymmetry: bool = Field(default=True, description="Extract asymmetry features")
    
    # Band definitions
    delta_band: List[float] = Field(default=[1, 4], description="Delta band frequencies")
    theta_band: List[float] = Field(default=[4, 8], description="Theta band frequencies")
    alpha_band: List[float] = Field(default=[8, 13], description="Alpha band frequencies")
    beta_band: List[float] = Field(default=[13, 30], description="Beta band frequencies")
    gamma_band: List[float] = Field(default=[30, 45], description="Gamma band frequencies")
    
    # Connectivity settings
    connectivity_method: str = Field(default="coherence", description="Connectivity method")
    connectivity_freq_bands: List[List[float]] = Field(
        default=[[8, 13], [13, 30]], 
        description="Frequency bands for connectivity analysis"
    )
    
    @validator('overlap')
    def validate_overlap(cls, v):
        if not 0.0 <= v < 1.0:
            raise ValueError('Overlap must be between 0.0 and 1.0')
        return v
    
    @validator('ica_components')
    def validate_ica_components(cls, v, values):
        if v is not None and 'channels' in values and v > len(values['channels']):
            raise ValueError('ICA components cannot exceed number of channels')
        return v
