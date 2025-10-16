"""
DreamWalk Web Dashboard

Real-time monitoring and visualization dashboard for the DreamWalk system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import redis.asyncio as redis
import httpx
import json
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response

from .models.dashboard_models import (
    DashboardData, SessionMetrics, ServiceStatus, WorldStateHistory,
    EEGSignalData, EmotionData, MotifData
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus metrics
DASHBOARD_VIEWS = Counter(
    'dashboard_views_total',
    'Total number of dashboard page views'
)

WEBSOCKET_CONNECTIONS = Gauge(
    'dashboard_websocket_connections_total',
    'Number of active WebSocket connections'
)

DATA_REQUESTS = Counter(
    'dashboard_data_requests_total',
    'Total number of data requests',
    ['data_type']
)

app = FastAPI(
    title="DreamWalk Web Dashboard",
    description="Real-time monitoring and visualization dashboard",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global state
redis_client: Optional[redis.Redis] = None
http_client: Optional[httpx.AsyncClient] = None
active_websockets: List[WebSocket] = []

# Service URLs
REALTIME_SERVER_URL = "http://realtime-server:8003"
SIGNAL_PROCESSOR_URL = "http://signal-processor:8001"
NEURAL_DECODER_URL = "http://neural-decoder:8002"
TEXTURE_GENERATOR_URL = "http://texture-generator:8005"
NARRATIVE_LAYER_URL = "http://narrative-layer:8006"


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global redis_client, http_client
    
    logger.info("Starting DreamWalk web dashboard...")
    
    # Initialize Redis connection
    redis_client = redis.from_url("redis://redis:6379", decode_responses=True)
    await redis_client.ping()
    
    # Initialize HTTP client
    http_client = httpx.AsyncClient(timeout=30.0)
    
    logger.info("Web dashboard started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if redis_client:
        await redis_client.close()
    if http_client:
        await http_client.aclose()
    logger.info("Web dashboard stopped")


@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page"""
    DASHBOARD_VIEWS.inc()
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/dashboard-data")
async def get_dashboard_data():
    """Get comprehensive dashboard data"""
    try:
        DATA_REQUESTS.labels(data_type="dashboard_data").inc()
        
        # Get data from all services
        dashboard_data = await _gather_dashboard_data()
        
        return dashboard_data
        
    except Exception as e:
        logger.error("Failed to get dashboard data", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions")
async def get_sessions():
    """Get all active sessions"""
    try:
        DATA_REQUESTS.labels(data_type="sessions").inc()
        
        async with http_client.get(f"{REALTIME_SERVER_URL}/sessions") as response:
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail="Failed to get sessions")
                
    except Exception as e:
        logger.error("Failed to get sessions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}/metrics")
async def get_session_metrics(session_id: str, duration: int = 300):
    """Get metrics for a specific session"""
    try:
        DATA_REQUESTS.labels(data_type="session_metrics").inc()
        
        # Get session data from Redis
        session_data = await redis_client.get(f"session:{session_id}")
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get recent world states
        world_states = await _get_session_world_states(session_id, duration)
        
        # Get EEG signal data if available
        eeg_data = await _get_session_eeg_data(session_id, duration)
        
        # Calculate metrics
        metrics = SessionMetrics(
            session_id=session_id,
            duration_seconds=duration,
            world_state_count=len(world_states),
            eeg_data_points=len(eeg_data),
            avg_emotion_valence=np.mean([ws.emotional_state.valence for ws in world_states]) if world_states else 0.0,
            avg_emotion_arousal=np.mean([ws.emotional_state.arousal for ws in world_states]) if world_states else 0.0,
            dominant_biome=_get_dominant_biome(world_states),
            active_motifs=_get_active_motifs(world_states)
        )
        
        return metrics
        
    except Exception as e:
        logger.error("Failed to get session metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/services/status")
async def get_services_status():
    """Get status of all services"""
    try:
        DATA_REQUESTS.labels(data_type="services_status").inc()
        
        services = {}
        
        # Check each service
        service_urls = {
            "realtime_server": REALTIME_SERVER_URL,
            "signal_processor": SIGNAL_PROCESSOR_URL,
            "neural_decoder": NEURAL_DECODER_URL,
            "texture_generator": TEXTURE_GENERATOR_URL,
            "narrative_layer": NARRATIVE_LAYER_URL
        }
        
        for service_name, url in service_urls.items():
            try:
                async with http_client.get(f"{url}/health", timeout=5.0) as response:
                    services[service_name] = ServiceStatus(
                        name=service_name,
                        url=url,
                        status="healthy" if response.status_code == 200 else "unhealthy",
                        response_time_ms=0,  # Could measure this
                        last_check=datetime.utcnow().isoformat()
                    )
            except Exception as e:
                services[service_name] = ServiceStatus(
                    name=service_name,
                    url=url,
                    status="unhealthy",
                    error=str(e),
                    last_check=datetime.utcnow().isoformat()
                )
        
        return {"services": services}
        
    except Exception as e:
        logger.error("Failed to get services status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/real-time/{session_id}")
async def get_realtime_data(session_id: str):
    """Get real-time data for a session"""
    try:
        DATA_REQUESTS.labels(data_type="realtime_data").inc()
        
        # Get latest world state
        world_state_data = await redis_client.get(f"world_state:{session_id}:latest")
        world_state = json.loads(world_state_data) if world_state_data else None
        
        # Get latest EEG features
        features_data = await redis_client.get(f"features:{session_id}:latest")
        features = json.loads(features_data) if features_data else None
        
        return {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "world_state": world_state,
            "features": features
        }
        
    except Exception as e:
        logger.error("Failed to get real-time data", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    await websocket.accept()
    active_websockets.append(websocket)
    WEBSOCKET_CONNECTIONS.set(len(active_websockets))
    
    try:
        while True:
            try:
                # Wait for client messages
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "subscribe_session":
                    session_id = message.get("session_id")
                    await _subscribe_to_session(websocket, session_id)
                
                elif message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                
            except WebSocketDisconnect:
                logger.info("Dashboard WebSocket disconnected")
                break
            except Exception as e:
                logger.error("Dashboard WebSocket error", error=str(e))
                
    except Exception as e:
        logger.error("Dashboard WebSocket connection error", error=str(e))
    finally:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
        WEBSOCKET_CONNECTIONS.set(len(active_websockets))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")


async def _gather_dashboard_data() -> DashboardData:
    """Gather comprehensive dashboard data"""
    try:
        # Get active sessions
        sessions_data = await redis_client.keys("session:*")
        active_sessions = len(sessions_data)
        
        # Get recent world states
        recent_world_states = []
        for key in await redis_client.keys("world_state:*:latest"):
            data = await redis_client.get(key)
            if data:
                recent_world_states.append(json.loads(data))
        
        # Get service statuses
        services_status = await _get_services_status()
        
        # Get system metrics
        system_metrics = await _get_system_metrics()
        
        return DashboardData(
            timestamp=datetime.utcnow().isoformat(),
            active_sessions=active_sessions,
            recent_world_states=recent_world_states[:10],  # Last 10
            services_status=services_status,
            system_metrics=system_metrics
        )
        
    except Exception as e:
        logger.error("Failed to gather dashboard data", error=str(e))
        return DashboardData(
            timestamp=datetime.utcnow().isoformat(),
            active_sessions=0,
            recent_world_states=[],
            services_status={},
            system_metrics={}
        )


async def _get_services_status() -> Dict[str, ServiceStatus]:
    """Get status of all services"""
    services = {}
    
    service_urls = {
        "realtime_server": REALTIME_SERVER_URL,
        "signal_processor": SIGNAL_PROCESSOR_URL,
        "neural_decoder": NEURAL_DECODER_URL,
        "texture_generator": TEXTURE_GENERATOR_URL,
        "narrative_layer": NARRATIVE_LAYER_URL
    }
    
    for service_name, url in service_urls.items():
        try:
            async with http_client.get(f"{url}/health", timeout=5.0) as response:
                services[service_name] = ServiceStatus(
                    name=service_name,
                    url=url,
                    status="healthy" if response.status_code == 200 else "unhealthy",
                    response_time_ms=0,
                    last_check=datetime.utcnow().isoformat()
                )
        except Exception as e:
            services[service_name] = ServiceStatus(
                name=service_name,
                url=url,
                status="unhealthy",
                error=str(e),
                last_check=datetime.utcnow().isoformat()
            )
    
    return services


async def _get_system_metrics() -> Dict[str, Any]:
    """Get system metrics from Redis"""
    try:
        # Get Redis info
        redis_info = await redis_client.info()
        
        # Get key counts
        session_keys = await redis_client.keys("session:*")
        world_state_keys = await redis_client.keys("world_state:*")
        feature_keys = await redis_client.keys("features:*")
        
        return {
            "redis_memory_usage": redis_info.get("used_memory_human", "0B"),
            "redis_connected_clients": redis_info.get("connected_clients", 0),
            "total_sessions": len(session_keys),
            "total_world_states": len(world_state_keys),
            "total_feature_entries": len(feature_keys)
        }
        
    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e))
        return {}


async def _get_session_world_states(session_id: str, duration: int) -> List[Dict[str, Any]]:
    """Get world states for a session within duration"""
    try:
        # Get session history from Redis
        history_data = await redis_client.lrange(f"session:{session_id}:history", 0, -1)
        
        world_states = []
        cutoff_time = datetime.utcnow() - timedelta(seconds=duration)
        
        for data in history_data:
            try:
                world_state = json.loads(data)
                state_time = datetime.fromisoformat(world_state.get("timestamp", ""))
                
                if state_time >= cutoff_time:
                    world_states.append(world_state)
                    
            except Exception as e:
                logger.warning("Failed to parse world state", error=str(e))
                continue
        
        return world_states
        
    except Exception as e:
        logger.error("Failed to get session world states", error=str(e))
        return []


async def _get_session_eeg_data(session_id: str, duration: int) -> List[EEGSignalData]:
    """Get EEG signal data for a session"""
    try:
        # This would typically come from the signal processor
        # For now, return empty list
        return []
        
    except Exception as e:
        logger.error("Failed to get session EEG data", error=str(e))
        return []


def _get_dominant_biome(world_states: List[Dict[str, Any]]) -> str:
    """Get the dominant biome from world states"""
    if not world_states:
        return "neutral"
    
    biome_counts = {}
    for state in world_states:
        biome = state.get("biome_type", "neutral")
        biome_counts[biome] = biome_counts.get(biome, 0) + 1
    
    return max(biome_counts.items(), key=lambda x: x[1])[0] if biome_counts else "neutral"


def _get_active_motifs(world_states: List[Dict[str, Any]]) -> List[str]:
    """Get active motifs from world states"""
    motifs = set()
    
    for state in world_states:
        state_motifs = state.get("motifs", [])
        for motif in state_motifs:
            if isinstance(motif, dict):
                motifs.add(motif.get("motif_type", ""))
            elif isinstance(motif, str):
                motifs.add(motif)
    
    return list(motifs)


async def _subscribe_to_session(websocket: WebSocket, session_id: str):
    """Subscribe to real-time updates for a session"""
    try:
        # This would set up real-time monitoring for the session
        # For now, just acknowledge the subscription
        await websocket.send_text(json.dumps({
            "type": "session_subscribed",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
    except Exception as e:
        logger.error("Failed to subscribe to session", error=str(e))


async def broadcast_to_dashboard(data: Dict[str, Any]):
    """Broadcast data to all connected dashboard WebSockets"""
    if not active_websockets:
        return
    
    message = json.dumps({
        "type": "dashboard_update",
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # Send to all connected clients
    disconnected = []
    for websocket in active_websockets:
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.warning("Failed to send to WebSocket", error=str(e))
            disconnected.append(websocket)
    
    # Remove disconnected WebSockets
    for websocket in disconnected:
        active_websockets.remove(websocket)
    
    WEBSOCKET_CONNECTIONS.set(len(active_websockets))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
