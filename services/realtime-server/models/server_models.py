"""
Server data models for DreamWalk real-time server
"""

from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import WebSocket


class StreamRequest(BaseModel):
    """Request to start a streaming session"""
    session_id: str = Field(..., description="Unique session identifier")
    signal_type: str = Field(default="mock", description="Signal source type (mock, eeg, fmri)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Signal stream configuration")


class StreamResponse(BaseModel):
    """Response after starting a streaming session"""
    status: str = Field(..., description="Session status")
    session_id: str = Field(..., description="Session identifier")
    websocket_url: str = Field(..., description="WebSocket URL for the session")


class WorldStateUpdate(BaseModel):
    """World state update broadcast to clients"""
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Update timestamp")
    world_state: Dict[str, Any] = Field(default_factory=dict, description="World state payload")


class SessionInfo(BaseModel):
    """Information about a streaming session"""
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Session status (active, stopped)")
    started_at: datetime = Field(..., description="Session start time")
    stopped_at: Optional[datetime] = Field(default=None, description="Session stop time")
    signal_type: str = Field(..., description="Signal source type")
    config: Dict[str, Any] = Field(default_factory=dict, description="Session configuration")


class ServerConfig(BaseModel):
    """Real-time server configuration"""
    processing_rate_hz: float = Field(default=10.0, description="World state processing rate in Hz")
    enable_texture_generation: bool = Field(default=True, description="Enable texture generation")
    enable_narrative: bool = Field(default=True, description="Enable narrative generation")


class ConnectionManager:
    """Manages active WebSocket connections per session"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        self.active_connections[session_id] = websocket

    async def disconnect(self, session_id: str) -> None:
        self.active_connections.pop(session_id, None)

    async def send_personal_message(self, message: Dict[str, Any], session_id: str) -> None:
        websocket = self.active_connections.get(session_id)
        if websocket is not None:
            await websocket.send_json(message)
