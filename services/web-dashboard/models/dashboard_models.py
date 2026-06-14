"""
Dashboard data models for DreamWalk
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    """Health status of a single backend service"""
    name: str = Field(..., description="Service name")
    url: str = Field(..., description="Base URL of the service")
    status: str = Field(..., description="Service status (healthy, unhealthy)")
    response_time_ms: float = Field(default=0.0, description="Last health check response time")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of last health check")
    error: Optional[str] = Field(default=None, description="Error message if unhealthy")


class DashboardData(BaseModel):
    """Aggregated data for the main dashboard view"""
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Snapshot timestamp")
    active_sessions: int = Field(default=0, description="Number of active sessions")
    recent_world_states: List[Dict[str, Any]] = Field(default_factory=list, description="Most recent world states")
    services_status: Dict[str, ServiceStatus] = Field(default_factory=dict, description="Status of backend services")
    system_metrics: Dict[str, Any] = Field(default_factory=dict, description="Redis/system level metrics")


class SessionMetrics(BaseModel):
    """Aggregated metrics for a single session"""
    session_id: str = Field(..., description="Session identifier")
    duration_seconds: int = Field(..., description="Duration over which metrics were computed")
    world_state_count: int = Field(default=0, description="Number of world states recorded")
    eeg_data_points: int = Field(default=0, description="Number of EEG data points recorded")
    avg_emotion_valence: float = Field(default=0.0, description="Average emotional valence")
    avg_emotion_arousal: float = Field(default=0.0, description="Average emotional arousal")
    dominant_biome: str = Field(default="neutral", description="Most frequent biome type")
    active_motifs: List[str] = Field(default_factory=list, description="Distinct active neural motifs")


class WorldStateHistory(BaseModel):
    """Historical world states for a session"""
    session_id: str = Field(..., description="Session identifier")
    world_states: List[Dict[str, Any]] = Field(default_factory=list, description="World state snapshots")
    timestamps: List[datetime] = Field(default_factory=list, description="Timestamps for each snapshot")


class EEGSignalData(BaseModel):
    """Raw EEG signal samples for dashboard display"""
    channel: str = Field(..., description="EEG channel name")
    values: List[float] = Field(default_factory=list, description="Signal samples")
    sampling_rate: float = Field(default=250.0, description="Sampling rate in Hz")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Sample window timestamp")


class EmotionData(BaseModel):
    """Emotional state summary for dashboard display"""
    valence: float = Field(default=0.0, description="Valence (-1 to 1)")
    arousal: float = Field(default=0.0, description="Arousal (0 to 1)")
    dominance: float = Field(default=0.0, description="Dominance (0 to 1)")
    dominant_emotion: str = Field(default="neutral", description="Dominant emotion category")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Estimation timestamp")


class MotifData(BaseModel):
    """Neural motif summary for dashboard display"""
    motif_type: str = Field(..., description="Type of detected motif")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Detection confidence")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")
