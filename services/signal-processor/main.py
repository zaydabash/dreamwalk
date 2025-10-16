"""
DreamWalk Signal Processor Service

Real-time EEG/fMRI signal processing with artifact removal, feature extraction,
and streaming to downstream services.
"""

import asyncio
import logging
from typing import AsyncGenerator, Dict, List, Optional, Tuple
from datetime import datetime
import structlog

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response

from .processors.eeg_processor import EEGProcessor
from .processors.fmri_processor import fMRIProcessor
from .feature_extractors.neural_features import NeuralFeatureExtractor
from .streamers.lsl_streamer import LSLStreamer
from .streamers.mock_streamer import MockStreamer
from .models.signal_models import SignalData, ProcessedFeatures, EEGConfig, fMRIConfig

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
SIGNAL_PROCESSING_DURATION = Histogram(
    'signal_processing_duration_seconds',
    'Time spent processing signals',
    ['signal_type', 'processing_stage']
)

FEATURES_EXTRACTED = Counter(
    'features_extracted_total',
    'Total number of feature vectors extracted',
    ['signal_type', 'feature_type']
)

ACTIVE_STREAMS = Gauge(
    'active_streams_total',
    'Number of active signal streams'
)

SIGNAL_QUALITY = Gauge(
    'signal_quality_score',
    'Quality score of incoming signals',
    ['stream_id']
)

app = FastAPI(
    title="DreamWalk Signal Processor",
    description="Real-time neural signal processing service",
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
eeg_processor: Optional[EEGProcessor] = None
fmri_processor: Optional[fMRIProcessor] = None
feature_extractor: Optional[NeuralFeatureExtractor] = None
active_streams: Dict[str, Dict] = {}


class StreamConfig(BaseModel):
    """Configuration for signal streaming"""
    signal_type: str = Field(..., description="Type of signal: eeg, fmri, or mock")
    config: Dict = Field(default_factory=dict, description="Signal-specific configuration")
    stream_id: str = Field(..., description="Unique identifier for this stream")


class ProcessingResult(BaseModel):
    """Result of signal processing"""
    stream_id: str
    timestamp: datetime
    features: ProcessedFeatures
    quality_score: float
    processing_time_ms: float


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global redis_client, eeg_processor, fmri_processor, feature_extractor
    
    # Initialize Redis connection
    redis_client = redis.from_url("redis://redis:6379", decode_responses=True)
    await redis_client.ping()
    
    # Initialize processors
    eeg_processor = EEGProcessor()
    fmri_processor = fMRIProcessor()
    feature_extractor = NeuralFeatureExtractor()
    
    logger.info("Signal processor service started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if redis_client:
        await redis_client.close()
    logger.info("Signal processor service stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")


@app.post("/streams/start")
async def start_stream(config: StreamConfig):
    """Start a new signal stream"""
    try:
        stream_id = config.stream_id
        
        # Initialize stream configuration
        stream_config = {
            "signal_type": config.signal_type,
            "config": config.config,
            "started_at": datetime.utcnow(),
            "status": "active"
        }
        
        active_streams[stream_id] = stream_config
        
        # Store in Redis for persistence
        await redis_client.hset(
            f"stream:{stream_id}",
            mapping=stream_config
        )
        
        ACTIVE_STREAMS.set(len(active_streams))
        
        logger.info("Started stream", stream_id=stream_id, signal_type=config.signal_type)
        
        return {"status": "started", "stream_id": stream_id}
        
    except Exception as e:
        logger.error("Failed to start stream", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/streams/stop/{stream_id}")
async def stop_stream(stream_id: str):
    """Stop a signal stream"""
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    # Update status
    active_streams[stream_id]["status"] = "stopped"
    active_streams[stream_id]["stopped_at"] = datetime.utcnow()
    
    # Update Redis
    await redis_client.hset(
        f"stream:{stream_id}",
        "status", "stopped",
        "stopped_at", datetime.utcnow().isoformat()
    )
    
    # Remove from active streams
    del active_streams[stream_id]
    ACTIVE_STREAMS.set(len(active_streams))
    
    logger.info("Stopped stream", stream_id=stream_id)
    
    return {"status": "stopped", "stream_id": stream_id}


@app.get("/streams")
async def list_streams():
    """List all active streams"""
    return {
        "active_streams": len(active_streams),
        "streams": list(active_streams.keys())
    }


@app.get("/streams/{stream_id}/status")
async def get_stream_status(stream_id: str):
    """Get status of a specific stream"""
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    return active_streams[stream_id]


@app.websocket("/streams/{stream_id}/process")
async def process_stream_websocket(websocket: WebSocket, stream_id: str):
    """WebSocket endpoint for real-time signal processing"""
    await websocket.accept()
    
    if stream_id not in active_streams:
        await websocket.close(code=1008, reason="Stream not found")
        return
    
    try:
        # Get stream configuration
        stream_config = active_streams[stream_id]
        signal_type = stream_config["signal_type"]
        
        # Initialize appropriate streamer
        if signal_type == "mock":
            streamer = MockStreamer(config=stream_config["config"])
        elif signal_type == "eeg":
            streamer = LSLStreamer(config=stream_config["config"])
        elif signal_type == "fmri":
            # fMRI is typically offline processing
            await websocket.close(code=1008, reason="fMRI processing not supported via WebSocket")
            return
        else:
            await websocket.close(code=1008, reason="Unsupported signal type")
            return
        
        # Process signals in real-time
        async for raw_data in streamer.stream():
            with SIGNAL_PROCESSING_DURATION.labels(
                signal_type=signal_type, 
                processing_stage="total"
            ).time():
                
                # Process the signal
                if signal_type == "eeg":
                    processed_data = await eeg_processor.process(raw_data)
                else:  # mock
                    processed_data = await eeg_processor.process(raw_data)
                
                # Extract features
                with SIGNAL_PROCESSING_DURATION.labels(
                    signal_type=signal_type,
                    processing_stage="feature_extraction"
                ).time():
                    features = await feature_extractor.extract(processed_data)
                
                # Calculate quality score
                quality_score = calculate_signal_quality(processed_data)
                SIGNAL_QUALITY.labels(stream_id=stream_id).set(quality_score)
                
                # Create result
                result = ProcessingResult(
                    stream_id=stream_id,
                    timestamp=datetime.utcnow(),
                    features=features,
                    quality_score=quality_score,
                    processing_time_ms=0.0  # Will be filled by metrics
                )
                
                # Update metrics
                FEATURES_EXTRACTED.labels(
                    signal_type=signal_type,
                    feature_type="neural_features"
                ).inc()
                
                # Send result via WebSocket
                await websocket.send_json(result.dict())
                
                # Store in Redis for other services
                await redis_client.setex(
                    f"features:{stream_id}:latest",
                    60,  # 1 minute TTL
                    result.json()
                )
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", stream_id=stream_id)
    except Exception as e:
        logger.error("Error in stream processing", stream_id=stream_id, error=str(e))
        await websocket.close(code=1011, reason="Internal error")


@app.post("/process/batch")
async def process_batch(file_path: str, signal_type: str):
    """Process a batch of signals from file"""
    try:
        # Load data
        if signal_type == "eeg":
            raw_data = pd.read_parquet(file_path)
            processed_data = await eeg_processor.process_batch(raw_data)
        elif signal_type == "fmri":
            processed_data = await fmri_processor.process_file(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported signal type")
        
        # Extract features
        features = await feature_extractor.extract_batch(processed_data)
        
        # Store results
        result_path = f"/app/data/processed_{signal_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
        features.to_parquet(result_path)
        
        return {
            "status": "completed",
            "result_path": result_path,
            "samples_processed": len(features)
        }
        
    except Exception as e:
        logger.error("Batch processing failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


def calculate_signal_quality(processed_data: SignalData) -> float:
    """Calculate signal quality score (0-1)"""
    try:
        # Simple quality metrics
        data = processed_data.data
        
        # Check for NaN values
        nan_ratio = np.isnan(data).sum() / data.size
        
        # Check signal amplitude (should be reasonable for EEG)
        amplitude_std = np.std(data)
        amplitude_score = 1.0 if 1e-6 < amplitude_std < 1e-3 else 0.5
        
        # Check for flat signals
        flat_score = 1.0 if np.std(data) > 1e-8 else 0.0
        
        # Overall quality score
        quality = (1 - nan_ratio) * amplitude_score * flat_score
        
        return max(0.0, min(1.0, quality))
        
    except Exception:
        return 0.0


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
