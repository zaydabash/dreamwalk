"""
DreamWalk Neural Decoder Service

Maps neural features to CLIP embeddings and emotional states for world generation.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import clip
from sentence_transformers import SentenceTransformer
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response

from .models.decoder_models import (
    DecoderRequest, DecoderResponse, EmotionalState, NeuralMotif, 
    WorldState, DecoderConfig
)
from .decoders.eeg_to_clip import EEGToCLIPDecoder
from .decoders.emotion_classifier import EmotionClassifier
from .decoders.motif_detector import MotifDetector
from .synthetic_data.generator import SyntheticDataGenerator

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
DECODING_DURATION = Histogram(
    'neural_decoding_duration_seconds',
    'Time spent decoding neural features',
    ['decoder_type', 'model_type']
)

EMOTIONS_PREDICTED = Counter(
    'emotions_predicted_total',
    'Total number of emotion predictions',
    ['emotion_type', 'confidence_level']
)

MOTIFS_DETECTED = Counter(
    'motifs_detected_total',
    'Total number of motifs detected',
    ['motif_type', 'confidence_level']
)

ACTIVE_DECODERS = Gauge(
    'active_decoders_total',
    'Number of active decoder instances'
)

MODEL_INFERENCE_TIME = Histogram(
    'model_inference_duration_seconds',
    'Model inference time',
    ['model_name', 'input_size']
)

app = FastAPI(
    title="DreamWalk Neural Decoder",
    description="Neural feature to semantic embedding decoder service",
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
eeg_decoder: Optional[EEGToCLIPDecoder] = None
emotion_classifier: Optional[EmotionClassifier] = None
motif_detector: Optional[MotifDetector] = None
clip_model = None
sentence_transformer = None
synthetic_generator: Optional[SyntheticDataGenerator] = None


@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global redis_client, eeg_decoder, emotion_classifier, motif_detector
    global clip_model, sentence_transformer, synthetic_generator
    
    logger.info("Starting neural decoder service...")
    
    # Initialize Redis connection
    redis_client = redis.from_url("redis://redis:6379", decode_responses=True)
    await redis_client.ping()
    
    # Initialize CLIP model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        logger.info("CLIP model loaded successfully", device=device)
    except Exception as e:
        logger.error("Failed to load CLIP model", error=str(e))
        raise
    
    # Initialize sentence transformer
    try:
        sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence transformer loaded successfully")
    except Exception as e:
        logger.error("Failed to load sentence transformer", error=str(e))
        raise
    
    # Initialize decoders
    try:
        eeg_decoder = EEGToCLIPDecoder(
            clip_model=clip_model,
            model_path="/app/models/checkpoints/eeg_to_clip.pth"
        )
        await eeg_decoder.load_model()
        
        emotion_classifier = EmotionClassifier(
            model_path="/app/models/checkpoints/emotion_classifier.pth"
        )
        await emotion_classifier.load_model()
        
        motif_detector = MotifDetector(
            model_path="/app/models/checkpoints/motif_detector.pth"
        )
        await motif_detector.load_model()
        
        logger.info("All decoders initialized successfully")
        
    except Exception as e:
        logger.warning("Failed to load pre-trained models, using synthetic training", error=str(e))
        # Initialize with synthetic data training
        synthetic_generator = SyntheticDataGenerator()
        await _initialize_with_synthetic_data()
    
    ACTIVE_DECODERS.set(3)  # EEG, emotion, motif decoders
    logger.info("Neural decoder service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if redis_client:
        await redis_client.close()
    logger.info("Neural decoder service stopped")


async def _initialize_with_synthetic_data():
    """Initialize models with synthetic data if pre-trained models are not available"""
    global eeg_decoder, emotion_classifier, motif_detector, synthetic_generator
    
    try:
        logger.info("Initializing models with synthetic data...")
        
        # Generate synthetic training data
        synthetic_data = await synthetic_generator.generate_training_data(
            n_samples=10000,
            n_channels=8
        )
        
        # Train EEG to CLIP decoder
        eeg_decoder = EEGToCLIPDecoder(clip_model=clip_model)
        await eeg_decoder.train_synthetic(synthetic_data)
        
        # Train emotion classifier
        emotion_classifier = EmotionClassifier()
        await emotion_classifier.train_synthetic(synthetic_data)
        
        # Train motif detector
        motif_detector = MotifDetector()
        await motif_detector.train_synthetic(synthetic_data)
        
        # Save models
        await eeg_decoder.save_model("/app/models/checkpoints/eeg_to_clip_synthetic.pth")
        await emotion_classifier.save_model("/app/models/checkpoints/emotion_classifier_synthetic.pth")
        await motif_detector.save_model("/app/models/checkpoints/motif_detector_synthetic.pth")
        
        logger.info("Models trained and saved with synthetic data")
        
    except Exception as e:
        logger.error("Failed to initialize with synthetic data", error=str(e))
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "models_loaded": {
            "eeg_decoder": eeg_decoder is not None,
            "emotion_classifier": emotion_classifier is not None,
            "motif_detector": motif_detector is not None,
            "clip_model": clip_model is not None
        }
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")


@app.post("/decode", response_model=DecoderResponse)
async def decode_neural_features(request: DecoderRequest):
    """Decode neural features to world generation parameters"""
    try:
        with DECODING_DURATION.labels(decoder_type="full", model_type="ensemble").time():
            
            # Decode to CLIP embeddings
            clip_embedding = await _decode_to_clip(request.features)
            
            # Classify emotions
            emotional_state = await _classify_emotions(request.features)
            
            # Detect motifs
            motifs = await _detect_motifs(request.features)
            
            # Generate semantic embedding
            semantic_embedding = await _generate_semantic_embedding(emotional_state, motifs)
            
            # Create world state
            world_state = await _create_world_state(
                emotional_state, motifs, clip_embedding, semantic_embedding, request.session_id
            )
            
            # Store results in Redis
            await _store_decoding_results(request.session_id, world_state)
            
            # Update metrics
            EMOTIONS_PREDICTED.labels(
                emotion_type=emotional_state.dominant_emotion,
                confidence_level=_get_confidence_level(emotional_state.confidence)
            ).inc()
            
            for motif in motifs:
                MOTIFS_DETECTED.labels(
                    motif_type=motif.motif_type,
                    confidence_level=_get_confidence_level(motif.confidence)
                ).inc()
            
            return DecoderResponse(
                session_id=request.session_id,
                timestamp="2024-01-01T00:00:00Z",
                world_state=world_state,
                processing_time_ms=0.0  # Will be filled by metrics
            )
            
    except Exception as e:
        logger.error("Decoding failed", error=str(e), session_id=request.session_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/decode/clip")
async def decode_to_clip(request: DecoderRequest):
    """Decode neural features to CLIP embeddings only"""
    try:
        with DECODING_DURATION.labels(decoder_type="clip", model_type="eeg_to_clip").time():
            clip_embedding = await _decode_to_clip(request.features)
            
            return {
                "session_id": request.session_id,
                "clip_embedding": clip_embedding,
                "embedding_dim": len(clip_embedding)
            }
            
    except Exception as e:
        logger.error("CLIP decoding failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/emotions")
async def classify_emotions(request: DecoderRequest):
    """Classify emotional state from neural features"""
    try:
        with DECODING_DURATION.labels(decoder_type="emotion", model_type="emotion_classifier").time():
            emotional_state = await _classify_emotions(request.features)
            
            return {
                "session_id": request.session_id,
                "emotional_state": emotional_state
            }
            
    except Exception as e:
        logger.error("Emotion classification failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/motifs")
async def detect_motifs(request: DecoderRequest):
    """Detect neural motifs from features"""
    try:
        with DECODING_DURATION.labels(decoder_type="motif", model_type="motif_detector").time():
            motifs = await _detect_motifs(request.features)
            
            return {
                "session_id": request.session_id,
                "motifs": motifs,
                "motif_count": len(motifs)
            }
            
    except Exception as e:
        logger.error("Motif detection failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/status")
async def get_models_status():
    """Get status of all models"""
    return {
        "eeg_decoder": {
            "loaded": eeg_decoder is not None,
            "model_path": eeg_decoder.model_path if eeg_decoder else None,
            "input_dim": eeg_decoder.input_dim if eeg_decoder else None,
            "output_dim": eeg_decoder.output_dim if eeg_decoder else None
        },
        "emotion_classifier": {
            "loaded": emotion_classifier is not None,
            "model_path": emotion_classifier.model_path if emotion_classifier else None,
            "supported_emotions": emotion_classifier.supported_emotions if emotion_classifier else []
        },
        "motif_detector": {
            "loaded": motif_detector is not None,
            "model_path": motif_detector.model_path if motif_detector else None,
            "supported_motifs": motif_detector.supported_motifs if motif_detector else []
        },
        "clip_model": {
            "loaded": clip_model is not None,
            "model_name": "ViT-B/32",
            "embedding_dim": 512
        }
    }


@app.post("/models/retrain")
async def retrain_models():
    """Retrain models with latest data"""
    try:
        if synthetic_generator is None:
            synthetic_generator = SyntheticDataGenerator()
        
        await _initialize_with_synthetic_data()
        
        return {"status": "retrained", "message": "Models retrained successfully"}
        
    except Exception as e:
        logger.error("Model retraining failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def _decode_to_clip(features) -> List[float]:
    """Decode neural features to CLIP embedding"""
    try:
        if eeg_decoder is None:
            # Return random embedding if model not available
            return np.random.normal(0, 1, 512).tolist()
        
        with MODEL_INFERENCE_TIME.labels(model_name="eeg_to_clip", input_size="variable").time():
            clip_embedding = await eeg_decoder.decode(features)
            return clip_embedding.tolist()
            
    except Exception as e:
        logger.error("CLIP decoding error", error=str(e))
        # Return neutral embedding on error
        return np.zeros(512).tolist()


async def _classify_emotions(features) -> EmotionalState:
    """Classify emotional state from neural features"""
    try:
        if emotion_classifier is None:
            # Return neutral state if model not available
            return EmotionalState(
                valence=0.0,
                arousal=0.5,
                dominance=0.5,
                confidence=0.0,
                dominant_emotion="neutral"
            )
        
        with MODEL_INFERENCE_TIME.labels(model_name="emotion_classifier", input_size="variable").time():
            emotional_state = await emotion_classifier.classify(features)
            return emotional_state
            
    except Exception as e:
        logger.error("Emotion classification error", error=str(e))
        return EmotionalState(
            valence=0.0,
            arousal=0.5,
            dominance=0.5,
            confidence=0.0,
            dominant_emotion="neutral"
        )


async def _detect_motifs(features) -> List[NeuralMotif]:
    """Detect neural motifs from features"""
    try:
        if motif_detector is None:
            return []
        
        with MODEL_INFERENCE_TIME.labels(model_name="motif_detector", input_size="variable").time():
            motifs = await motif_detector.detect(features)
            return motifs
            
    except Exception as e:
        logger.error("Motif detection error", error=str(e))
        return []


async def _generate_semantic_embedding(emotional_state: EmotionalState, motifs: List[NeuralMotif]) -> List[float]:
    """Generate semantic embedding from emotional state and motifs"""
    try:
        if sentence_transformer is None:
            return np.random.normal(0, 1, 384).tolist()
        
        # Create text description
        description = f"Emotional state: {emotional_state.dominant_emotion}, "
        description += f"valence: {emotional_state.valence:.2f}, "
        description += f"arousal: {emotional_state.arousal:.2f}"
        
        if motifs:
            description += f", motifs: {', '.join([m.motif_type for m in motifs])}"
        
        # Generate embedding
        embedding = sentence_transformer.encode([description])[0]
        return embedding.tolist()
        
    except Exception as e:
        logger.error("Semantic embedding generation error", error=str(e))
        return np.random.normal(0, 1, 384).tolist()


async def _create_world_state(emotional_state: EmotionalState, motifs: List[NeuralMotif],
                            clip_embedding: List[float], semantic_embedding: List[float],
                            session_id: str) -> WorldState:
    """Create world state from decoded features"""
    try:
        # Map emotional state to world parameters
        biome_type = _map_emotion_to_biome(emotional_state)
        weather_intensity = _map_arousal_to_weather(emotional_state.arousal)
        lighting_mood = _map_valence_to_lighting(emotional_state.valence)
        color_palette = _map_emotion_to_colors(emotional_state)
        
        # Adjust parameters based on motifs
        for motif in motifs:
            if motif.confidence > 0.7:  # High confidence motifs
                biome_type, weather_intensity, lighting_mood = _adjust_for_motif(
                    biome_type, weather_intensity, lighting_mood, motif
                )
        
        return WorldState(
            emotional_state=emotional_state,
            motifs=motifs,
            clip_embedding=clip_embedding,
            semantic_embedding=semantic_embedding,
            biome_type=biome_type,
            weather_intensity=weather_intensity,
            lighting_mood=lighting_mood,
            color_palette=color_palette,
            object_density=_calculate_object_density(emotional_state),
            structure_level=_calculate_structure_level(emotional_state),
            ambient_volume=_calculate_ambient_volume(emotional_state),
            music_intensity=_calculate_music_intensity(emotional_state),
            sound_effects=_generate_sound_effects(emotional_state, motifs),
            change_rate=_calculate_change_rate(emotional_state),
            morph_speed=_calculate_morph_speed(emotional_state),
            session_id=session_id
        )
        
    except Exception as e:
        logger.error("World state creation error", error=str(e))
        # Return default world state
        return WorldState(
            emotional_state=emotional_state,
            motifs=motifs,
            clip_embedding=clip_embedding,
            semantic_embedding=semantic_embedding,
            biome_type="neutral",
            weather_intensity=0.5,
            lighting_mood="neutral",
            color_palette=[0.5, 0.5, 0.5],
            object_density=0.5,
            structure_level=0.5,
            ambient_volume=0.5,
            music_intensity=0.3,
            sound_effects=[],
            change_rate=0.1,
            morph_speed=1.0,
            session_id=session_id
        )


def _map_emotion_to_biome(emotional_state: EmotionalState) -> str:
    """Map emotional state to biome type"""
    if emotional_state.valence > 0.3 and emotional_state.arousal > 0.5:
        return "lush_forest"
    elif emotional_state.valence > 0.3 and emotional_state.arousal < 0.5:
        return "peaceful_garden"
    elif emotional_state.valence < -0.3 and emotional_state.arousal > 0.5:
        return "stormy_mountains"
    elif emotional_state.valence < -0.3 and emotional_state.arousal < 0.5:
        return "desert_wasteland"
    else:
        return "neutral_plains"


def _map_arousal_to_weather(arousal: float) -> float:
    """Map arousal level to weather intensity"""
    return min(1.0, max(0.0, arousal * 1.5 - 0.25))


def _map_valence_to_lighting(valence: float) -> str:
    """Map valence to lighting mood"""
    if valence > 0.3:
        return "bright_warm"
    elif valence < -0.3:
        return "dim_cool"
    else:
        return "neutral"


def _map_emotion_to_colors(emotional_state: EmotionalState) -> List[float]:
    """Map emotional state to RGB color palette"""
    # Base colors: [R, G, B] normalized to 0-1
    if emotional_state.valence > 0.3:
        # Positive valence: warm colors
        return [0.8, 0.6, 0.2]  # Warm gold
    elif emotional_state.valence < -0.3:
        # Negative valence: cool colors
        return [0.2, 0.4, 0.8]  # Cool blue
    else:
        # Neutral: balanced colors
        return [0.5, 0.5, 0.5]  # Neutral gray


def _adjust_for_motif(biome_type: str, weather_intensity: float, lighting_mood: str, 
                     motif: NeuralMotif) -> Tuple[str, float, str]:
    """Adjust world parameters based on detected motif"""
    if motif.motif_type == "meditation":
        return "zen_garden", max(0.1, weather_intensity - 0.3), "soft_warm"
    elif motif.motif_type == "stress":
        return "chaotic_void", min(1.0, weather_intensity + 0.3), "harsh_cool"
    elif motif.motif_type == "focus":
        return "crystal_caverns", weather_intensity, "sharp_neutral"
    elif motif.motif_type == "creativity":
        return "surreal_landscape", weather_intensity, "vibrant_changing"
    else:
        return biome_type, weather_intensity, lighting_mood


def _calculate_object_density(emotional_state: EmotionalState) -> float:
    """Calculate object placement density"""
    # Higher arousal = more objects, higher valence = more organized placement
    base_density = 0.3 + emotional_state.arousal * 0.4
    organization_factor = 0.5 + emotional_state.valence * 0.3
    return min(1.0, base_density * organization_factor)


def _calculate_structure_level(emotional_state: EmotionalState) -> float:
    """Calculate level of structural organization"""
    # Higher valence = more structure, higher arousal = more chaos
    base_structure = 0.5 + emotional_state.valence * 0.3
    chaos_factor = emotional_state.arousal * 0.2
    return max(0.0, min(1.0, base_structure - chaos_factor))


def _calculate_ambient_volume(emotional_state: EmotionalState) -> float:
    """Calculate ambient audio volume"""
    return 0.3 + emotional_state.arousal * 0.4


def _calculate_music_intensity(emotional_state: EmotionalState) -> float:
    """Calculate music intensity"""
    return 0.2 + emotional_state.arousal * 0.5


def _generate_sound_effects(emotional_state: EmotionalState, motifs: List[NeuralMotif]) -> List[str]:
    """Generate appropriate sound effects"""
    effects = []
    
    # Base effects from emotional state
    if emotional_state.valence > 0.3:
        effects.extend(["birds_chirping", "gentle_wind"])
    elif emotional_state.valence < -0.3:
        effects.extend(["distant_thunder", "howling_wind"])
    
    if emotional_state.arousal > 0.7:
        effects.extend(["intense_ambience", "dynamic_reverb"])
    
    # Motif-specific effects
    for motif in motifs:
        if motif.confidence > 0.6:
            if motif.motif_type == "meditation":
                effects.extend(["singing_bowls", "soft_chimes"])
            elif motif.motif_type == "stress":
                effects.extend(["mechanical_hum", "unsettling_drone"])
            elif motif.motif_type == "creativity":
                effects.extend(["ethereal_voices", "magical_sparkles"])
    
    return list(set(effects))  # Remove duplicates


def _calculate_change_rate(emotional_state: EmotionalState) -> float:
    """Calculate rate of environmental changes"""
    return 0.1 + emotional_state.arousal * 0.3


def _calculate_morph_speed(emotional_state: EmotionalState) -> float:
    """Calculate world morphing speed"""
    return 0.5 + emotional_state.arousal * 0.5


def _get_confidence_level(confidence: float) -> str:
    """Categorize confidence level"""
    if confidence > 0.8:
        return "high"
    elif confidence > 0.5:
        return "medium"
    else:
        return "low"


async def _store_decoding_results(session_id: str, world_state: WorldState):
    """Store decoding results in Redis"""
    try:
        await redis_client.setex(
            f"decoding:{session_id}:latest",
            300,  # 5 minute TTL
            world_state.json()
        )
        
        # Store in session history
        await redis_client.lpush(
            f"session:{session_id}:history",
            world_state.json()
        )
        await redis_client.ltrim(f"session:{session_id}:history", 0, 99)  # Keep last 100
        
    except Exception as e:
        logger.error("Failed to store decoding results", error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
