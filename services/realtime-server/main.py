"""
DreamWalk Real-time Server

Orchestrates real-time neural signal processing and world state generation.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as redis
import httpx
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response

from .models.server_models import (
    StreamRequest, StreamResponse, WorldStateUpdate, SessionInfo, 
    ConnectionManager, ServerConfig
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
ACTIVE_SESSIONS = Gauge(
    'active_sessions_total',
    'Number of active streaming sessions'
)

STREAM_MESSAGES_SENT = Counter(
    'stream_messages_sent_total',
    'Total number of stream messages sent',
    ['session_id']
)

STREAM_PROCESSING_DURATION = Histogram(
    'stream_processing_duration_seconds',
    'Time spent processing stream data',
    ['processing_stage']
)

SERVICE_CALLS = Counter(
    'service_calls_total',
    'Total number of service calls',
    ['service_name', 'endpoint', 'status']
)

app = FastAPI(
    title="DreamWalk Real-time Server",
    description="Real-time neural signal processing and world state orchestration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
redis_client: Optional[redis.Redis] = None
connection_manager = ConnectionManager()
http_client: Optional[httpx.AsyncClient] = None
config = ServerConfig()

# Service URLs
SIGNAL_PROCESSOR_URL = "http://signal-processor:8001"
NEURAL_DECODER_URL = "http://neural-decoder:8002"
TEXTURE_GENERATOR_URL = "http://texture-generator:8005"
NARRATIVE_LAYER_URL = "http://narrative-layer:8006"


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global redis_client, http_client
    
    logger.info("Starting DreamWalk real-time server...")
    
    # Initialize Redis connection
    redis_client = redis.from_url("redis://redis:6379", decode_responses=True)
    await redis_client.ping()
    
    # Initialize HTTP client
    http_client = httpx.AsyncClient(timeout=30.0)
    
    logger.info("Real-time server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if redis_client:
        await redis_client.close()
    if http_client:
        await http_client.aclose()
    logger.info("Real-time server stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_sessions": len(connection_manager.active_connections),
        "services": {
            "signal_processor": await _check_service_health(SIGNAL_PROCESSOR_URL),
            "neural_decoder": await _check_service_health(NEURAL_DECODER_URL),
            "texture_generator": await _check_service_health(TEXTURE_GENERATOR_URL),
            "narrative_layer": await _check_service_health(NARRATIVE_LAYER_URL)
        }
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")


@app.post("/sessions/start")
async def start_session(request: StreamRequest):
    """Start a new streaming session"""
    try:
        session_id = request.session_id
        
        # Check if session already exists
        if session_id in connection_manager.active_connections:
            raise HTTPException(status_code=400, detail="Session already active")
        
        # Initialize session in Redis
        session_info = SessionInfo(
            session_id=session_id,
            status="active",
            started_at=datetime.utcnow(),
            signal_type=request.signal_type,
            config=request.config
        )
        
        await redis_client.setex(
            f"session:{session_id}",
            3600,  # 1 hour TTL
            session_info.json()
        )
        
        # Start signal processing stream
        await _start_signal_processing(session_id, request)
        
        ACTIVE_SESSIONS.set(len(connection_manager.active_connections))
        
        logger.info("Session started", session_id=session_id)
        
        return {
            "status": "started",
            "session_id": session_id,
            "websocket_url": f"/ws/{session_id}"
        }
        
    except Exception as e:
        logger.error("Failed to start session", error=str(e), session_id=request.session_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/stop/{session_id}")
async def stop_session(session_id: str):
    """Stop a streaming session"""
    try:
        # Disconnect WebSocket if active
        if session_id in connection_manager.active_connections:
            await connection_manager.disconnect(session_id)
        
        # Update session status in Redis
        session_data = await redis_client.get(f"session:{session_id}")
        if session_data:
            session_info = SessionInfo.parse_raw(session_data)
            session_info.status = "stopped"
            session_info.stopped_at = datetime.utcnow()
            
            await redis_client.setex(
                f"session:{session_id}",
                3600,
                session_info.json()
            )
        
        ACTIVE_SESSIONS.set(len(connection_manager.active_connections))
        
        logger.info("Session stopped", session_id=session_id)
        
        return {"status": "stopped", "session_id": session_id}
        
    except Exception as e:
        logger.error("Failed to stop session", error=str(e), session_id=session_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Get session status"""
    try:
        session_data = await redis_client.get(f"session:{session_id}")
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_info = SessionInfo.parse_raw(session_data)
        
        # Add WebSocket connection status
        is_connected = session_id in connection_manager.active_connections
        
        return {
            "session_info": session_info,
            "websocket_connected": is_connected,
            "connection_count": len(connection_manager.active_connections)
        }
        
    except Exception as e:
        logger.error("Failed to get session status", error=str(e), session_id=session_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions():
    """List all sessions"""
    try:
        # Get all session keys
        session_keys = await redis_client.keys("session:*")
        sessions = []
        
        for key in session_keys:
            session_data = await redis_client.get(key)
            if session_data:
                session_info = SessionInfo.parse_raw(session_data)
                sessions.append(session_info)
        
        return {
            "sessions": sessions,
            "total_count": len(sessions),
            "active_count": len(connection_manager.active_connections)
        }
        
    except Exception as e:
        logger.error("Failed to list sessions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time streaming"""
    await connection_manager.connect(websocket, session_id)
    
    try:
        # Send initial connection confirmation
        await connection_manager.send_personal_message({
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }, session_id)
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "ping":
                    await connection_manager.send_personal_message({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }, session_id)
                
                elif message.get("type") == "request_world_update":
                    # Trigger world state update
                    await _trigger_world_update(session_id)
                
                elif message.get("type") == "set_manual_state":
                    # Set manual world state
                    await _set_manual_world_state(session_id, message.get("world_state"))
                
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected", session_id=session_id)
                break
            except Exception as e:
                logger.error("WebSocket error", error=str(e), session_id=session_id)
                await connection_manager.send_personal_message({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }, session_id)
                
    except Exception as e:
        logger.error("WebSocket connection error", error=str(e), session_id=session_id)
    finally:
        connection_manager.disconnect(session_id)


@app.post("/trigger/world-update/{session_id}")
async def trigger_world_update(session_id: str, background_tasks: BackgroundTasks):
    """Manually trigger world state update"""
    try:
        background_tasks.add_task(_trigger_world_update, session_id)
        
        return {
            "status": "triggered",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to trigger world update", error=str(e), session_id=session_id)
        raise HTTPException(status_code=500, detail=str(e))


async def _start_signal_processing(session_id: str, request: StreamRequest):
    """Start signal processing for a session"""
    try:
        # Configure signal processor stream
        stream_config = {
            "signal_type": request.signal_type,
            "config": request.config,
            "stream_id": session_id
        }
        
        # Start stream with signal processor
        async with http_client.post(
            f"{SIGNAL_PROCESSOR_URL}/streams/start",
            json=stream_config
        ) as response:
            if response.status_code != 200:
                raise Exception(f"Failed to start signal processing: {response.text}")
        
        # Start background processing loop
        asyncio.create_task(_process_signal_stream(session_id))
        
        SERVICE_CALLS.labels(
            service_name="signal_processor",
            endpoint="start_stream",
            status="success"
        ).inc()
        
    except Exception as e:
        logger.error("Failed to start signal processing", error=str(e), session_id=session_id)
        SERVICE_CALLS.labels(
            service_name="signal_processor",
            endpoint="start_stream",
            status="error"
        ).inc()
        raise


async def _process_signal_stream(session_id: str):
    """Process signal stream and generate world states"""
    try:
        while session_id in connection_manager.active_connections:
            with STREAM_PROCESSING_DURATION.labels(processing_stage="signal_processing").time():
                
                # Get latest features from signal processor
                features_data = await redis_client.get(f"features:{session_id}:latest")
                if not features_data:
                    await asyncio.sleep(0.1)  # Wait for features
                    continue
                
                features = json.loads(features_data)
                
                # Decode features to world state
                world_state = await _decode_to_world_state(session_id, features)
                
                # Store world state
                await redis_client.setex(
                    f"world_state:{session_id}:latest",
                    60,  # 1 minute TTL
                    world_state.json()
                )
                
                # Send to WebSocket clients
                await connection_manager.send_personal_message({
                    "type": "world_state_update",
                    "world_state": world_state.dict(),
                    "timestamp": datetime.utcnow().isoformat()
                }, session_id)
                
                STREAM_MESSAGES_SENT.labels(session_id=session_id).inc()
            
            # Rate limiting
            await asyncio.sleep(1.0 / config.processing_rate_hz)
            
    except Exception as e:
        logger.error("Signal stream processing error", error=str(e), session_id=session_id)
    finally:
        # Cleanup
        await _stop_signal_processing(session_id)


async def _decode_to_world_state(session_id: str, features: Dict[str, Any]) -> WorldStateUpdate:
    """Decode neural features to world state"""
    try:
        with STREAM_PROCESSING_DURATION.labels(processing_stage="neural_decoding").time():
            
            # Call neural decoder
            decoder_request = {
                "session_id": session_id,
                "features": features,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with http_client.post(
                f"{NEURAL_DECODER_URL}/decode",
                json=decoder_request
            ) as response:
                if response.status_code == 200:
                    decoder_response = response.json()
                    world_state_data = decoder_response["world_state"]
                    
                    SERVICE_CALLS.labels(
                        service_name="neural_decoder",
                        endpoint="decode",
                        status="success"
                    ).inc()
                    
                else:
                    raise Exception(f"Neural decoder failed: {response.text}")
        
        # Generate textures if needed
        if config.enable_texture_generation:
            await _generate_textures(session_id, world_state_data)
        
        # Generate narrative if enabled
        if config.enable_narrative:
            narrative = await _generate_narrative(session_id, world_state_data)
            world_state_data["narrative"] = narrative
        
        return WorldStateUpdate(**world_state_data)
        
    except Exception as e:
        logger.error("World state decoding failed", error=str(e), session_id=session_id)
        
        SERVICE_CALLS.labels(
            service_name="neural_decoder",
            endpoint="decode",
            status="error"
        ).inc()
        
        # Return default world state
        return WorldStateUpdate(
            session_id=session_id,
            timestamp=datetime.utcnow(),
            world_state={
                "biome_type": "neutral",
                "weather_intensity": 0.5,
                "lighting_mood": "neutral",
                "color_palette": [0.5, 0.5, 0.5],
                "object_density": 0.5,
                "structure_level": 0.5,
                "ambient_volume": 0.5,
                "music_intensity": 0.3,
                "sound_effects": [],
                "change_rate": 0.1,
                "morph_speed": 1.0
            }
        )


async def _generate_textures(session_id: str, world_state: Dict[str, Any]):
    """Generate textures for world state"""
    try:
        texture_request = {
            "session_id": session_id,
            "world_state": world_state,
            "texture_types": ["skybox", "terrain", "ambient"]
        }
        
        async with http_client.post(
            f"{TEXTURE_GENERATOR_URL}/generate",
            json=texture_request,
            timeout=30.0
        ) as response:
            if response.status_code == 200:
                SERVICE_CALLS.labels(
                    service_name="texture_generator",
                    endpoint="generate",
                    status="success"
                ).inc()
            else:
                SERVICE_CALLS.labels(
                    service_name="texture_generator",
                    endpoint="generate",
                    status="error"
                ).inc()
                
    except Exception as e:
        logger.error("Texture generation failed", error=str(e), session_id=session_id)
        SERVICE_CALLS.labels(
            service_name="texture_generator",
            endpoint="generate",
            status="error"
        ).inc()


async def _generate_narrative(session_id: str, world_state: Dict[str, Any]) -> str:
    """Generate narrative for world state"""
    try:
        narrative_request = {
            "session_id": session_id,
            "world_state": world_state,
            "narrative_type": "ambient"
        }
        
        async with http_client.post(
            f"{NARRATIVE_LAYER_URL}/generate",
            json=narrative_request,
            timeout=10.0
        ) as response:
            if response.status_code == 200:
                narrative_data = response.json()
                SERVICE_CALLS.labels(
                    service_name="narrative_layer",
                    endpoint="generate",
                    status="success"
                ).inc()
                return narrative_data.get("narrative", "")
            else:
                SERVICE_CALLS.labels(
                    service_name="narrative_layer",
                    endpoint="generate",
                    status="error"
                ).inc()
                return ""
                
    except Exception as e:
        logger.error("Narrative generation failed", error=str(e), session_id=session_id)
        SERVICE_CALLS.labels(
            service_name="narrative_layer",
            endpoint="generate",
            status="error"
        ).inc()
        return ""


async def _trigger_world_update(session_id: str):
    """Manually trigger world state update"""
    try:
        # Get latest features
        features_data = await redis_client.get(f"features:{session_id}:latest")
        if features_data:
            features = json.loads(features_data)
            world_state = await _decode_to_world_state(session_id, features)
            
            # Send update to WebSocket
            await connection_manager.send_personal_message({
                "type": "world_state_update",
                "world_state": world_state.dict(),
                "timestamp": datetime.utcnow().isoformat()
            }, session_id)
            
    except Exception as e:
        logger.error("Manual world update failed", error=str(e), session_id=session_id)


async def _set_manual_world_state(session_id: str, world_state_data: Dict[str, Any]):
    """Set manual world state"""
    try:
        world_state = WorldStateUpdate(
            session_id=session_id,
            timestamp=datetime.utcnow(),
            world_state=world_state_data
        )
        
        # Store in Redis
        await redis_client.setex(
            f"world_state:{session_id}:latest",
            60,
            world_state.json()
        )
        
        # Send to WebSocket
        await connection_manager.send_personal_message({
            "type": "world_state_update",
            "world_state": world_state.dict(),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "manual"
        }, session_id)
        
    except Exception as e:
        logger.error("Manual world state setting failed", error=str(e), session_id=session_id)


async def _stop_signal_processing(session_id: str):
    """Stop signal processing for a session"""
    try:
        async with http_client.post(
            f"{SIGNAL_PROCESSOR_URL}/streams/stop/{session_id}"
        ) as response:
            if response.status_code == 200:
                SERVICE_CALLS.labels(
                    service_name="signal_processor",
                    endpoint="stop_stream",
                    status="success"
                ).inc()
            else:
                SERVICE_CALLS.labels(
                    service_name="signal_processor",
                    endpoint="stop_stream",
                    status="error"
                ).inc()
                
    except Exception as e:
        logger.error("Failed to stop signal processing", error=str(e), session_id=session_id)


async def _check_service_health(service_url: str) -> bool:
    """Check if a service is healthy"""
    try:
        async with http_client.get(f"{service_url}/health", timeout=5.0) as response:
            return response.status_code == 200
    except Exception:
        return False


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
