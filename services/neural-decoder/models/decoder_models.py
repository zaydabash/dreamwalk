"""
Decoder data models for DreamWalk
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
import numpy as np


class DecoderRequest(BaseModel):
    """Request for neural decoding"""
    session_id: str = Field(..., description="Unique session identifier")
    features: Dict[str, Any] = Field(..., description="Neural features to decode")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Request timestamp")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Decoding configuration")


class DecoderResponse(BaseModel):
    """Response from neural decoding"""
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    world_state: "WorldState" = Field(..., description="Generated world state")
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall decoding confidence")


class EmotionalState(BaseModel):
    """Estimated emotional state from neural data"""
    valence: float = Field(..., ge=-1.0, le=1.0, description="Valence (-1 to 1)")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Arousal (0 to 1)")
    dominance: float = Field(..., ge=0.0, le=1.0, description="Dominance (0 to 1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Estimation confidence")
    dominant_emotion: str = Field(..., description="Dominant emotion category")
    emotion_scores: Dict[str, float] = Field(default_factory=dict, description="Scores for all emotion categories")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Estimation timestamp")
    
    @validator('dominant_emotion')
    def validate_dominant_emotion(cls, v):
        valid_emotions = [
            'neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised', 'disgusted',
            'relaxed', 'stressed', 'excited', 'calm', 'focused', 'confused'
        ]
        if v not in valid_emotions:
            raise ValueError(f'Invalid emotion: {v}. Must be one of {valid_emotions}')
        return v


class NeuralMotif(BaseModel):
    """Detected neural motif/pattern"""
    motif_id: str = Field(..., description="Unique motif identifier")
    motif_type: str = Field(..., description="Type of motif")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    features: Dict[str, float] = Field(..., description="Feature values that define this motif")
    description: str = Field(default="", description="Human-readable description")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")
    
    @validator('motif_type')
    def validate_motif_type(cls, v):
        valid_motifs = [
            'meditation', 'stress', 'focus', 'creativity', 'relaxation',
            'alertness', 'fatigue', 'confusion', 'flow_state', 'anxiety'
        ]
        if v not in valid_motifs:
            raise ValueError(f'Invalid motif type: {v}. Must be one of {valid_motifs}')
        return v


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
    
    @validator('color_palette')
    def validate_color_palette(cls, v):
        if len(v) != 3:
            raise ValueError('Color palette must have exactly 3 RGB values')
        for color in v:
            if not 0.0 <= color <= 1.0:
                raise ValueError('Color values must be between 0.0 and 1.0')
        return v


class DecoderConfig(BaseModel):
    """Configuration for neural decoders"""
    # Model settings
    model_path: str = Field(default="/app/models/checkpoints", description="Path to model checkpoints")
    device: str = Field(default="auto", description="Device for inference (auto, cpu, cuda)")
    batch_size: int = Field(default=1, description="Batch size for inference")
    
    # Feature processing
    normalize_features: bool = Field(default=True, description="Normalize input features")
    feature_scaling: str = Field(default="standard", description="Feature scaling method")
    
    # Confidence thresholds
    min_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")
    high_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="High confidence threshold")
    
    # Ensemble settings
    use_ensemble: bool = Field(default=False, description="Use ensemble of models")
    ensemble_weights: List[float] = Field(default_factory=list, description="Weights for ensemble models")
    
    # Caching
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl: int = Field(default=300, description="Cache TTL in seconds")
    
    @validator('feature_scaling')
    def validate_feature_scaling(cls, v):
        valid_methods = ['standard', 'minmax', 'robust', 'none']
        if v not in valid_methods:
            raise ValueError(f'Invalid scaling method: {v}. Must be one of {valid_methods}')
        return v


class TrainingData(BaseModel):
    """Training data for neural decoders"""
    features: List[Dict[str, Any]] = Field(..., description="Neural features")
    targets: List[Dict[str, Any]] = Field(..., description="Target values/labels")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Training metadata")
    session_ids: List[str] = Field(default_factory=list, description="Session identifiers")


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    model_name: str = Field(..., description="Model name")
    metric_type: str = Field(..., description="Type of metric")
    value: float = Field(..., description="Metric value")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metric timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metric metadata")
    
    @validator('metric_type')
    def validate_metric_type(cls, v):
        valid_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'auc',
            'mse', 'mae', 'r2_score', 'cosine_similarity'
        ]
        if v not in valid_metrics:
            raise ValueError(f'Invalid metric type: {v}. Must be one of {valid_metrics}')
        return v


class InferenceRequest(BaseModel):
    """Request for model inference"""
    model_type: str = Field(..., description="Type of model (eeg_to_clip, emotion_classifier, motif_detector)")
    input_data: Dict[str, Any] = Field(..., description="Input data for inference")
    session_id: str = Field(..., description="Session identifier")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Inference configuration")


class InferenceResponse(BaseModel):
    """Response from model inference"""
    model_type: str = Field(..., description="Model type")
    predictions: Dict[str, Any] = Field(..., description="Model predictions")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Prediction confidence")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


# Update forward references
DecoderResponse.model_rebuild()
