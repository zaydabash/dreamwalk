"""
Narrative Layer Service for DreamWalk

Generates ambient narration and symbolic descriptions based on neural state.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional
import json

import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DreamWalk Narrative Layer", version="1.0.0")

# Redis connection
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

class NeuralState(BaseModel):
    """Neural state data from the decoder"""
    valence: float  # -1 to 1
    arousal: float  # 0 to 1
    dominance: float  # -1 to 1
    motif_tags: List[str]
    latent_vector: List[float]

class NarrativeRequest(BaseModel):
    """Request for narrative generation"""
    neural_state: NeuralState
    context: Optional[str] = None
    length: str = "short"  # short, medium, long

class NarrativeResponse(BaseModel):
    """Generated narrative response"""
    ambient_text: str
    symbolic_objects: List[Dict[str, str]]
    mood_description: str
    confidence: float

def generate_ambient_narration(neural_state: NeuralState) -> str:
    """Generate ambient narration based on neural state"""
    
    # Map valence to emotional tone
    if neural_state.valence > 0.5:
        tone = "uplifting and hopeful"
    elif neural_state.valence > 0:
        tone = "gentle and peaceful"
    elif neural_state.valence > -0.5:
        tone = "melancholic and introspective"
    else:
        tone = "dark and mysterious"
    
    # Map arousal to energy level
    if neural_state.arousal > 0.7:
        energy = "dynamic and energetic"
    elif neural_state.arousal > 0.4:
        energy = "moderate and flowing"
    else:
        energy = "calm and serene"
    
    # Generate narrative based on motifs
    motif_text = ""
    if neural_state.motif_tags:
        motif_text = f" The air carries whispers of {', '.join(neural_state.motif_tags[:3])}."
    
    ambient_text = f"The dreamscape unfolds with {tone} energy, {energy} in its essence.{motif_text} The world responds to your inner state, shifting and morphing with each passing moment."
    
    return ambient_text

def generate_symbolic_objects(neural_state: NeuralState) -> List[Dict[str, str]]:
    """Generate symbolic object descriptions"""
    
    objects = []
    
    # Valence-based objects
    if neural_state.valence > 0.3:
        objects.append({
            "name": "Glowing Bridge",
            "description": "A radiant bridge spanning across the void, representing hope and connection",
            "symbolism": "Hope, transition, positive change"
        })
    elif neural_state.valence < -0.3:
        objects.append({
            "name": "Crumbling Tower",
            "description": "An ancient tower slowly dissolving into mist, symbolizing uncertainty",
            "symbolism": "Anxiety, instability, emotional turmoil"
        })
    
    # Arousal-based objects
    if neural_state.arousal > 0.6:
        objects.append({
            "name": "Storm Clouds",
            "description": "Dynamic clouds swirling overhead, pulsing with energy",
            "symbolism": "High energy, excitement, intense emotions"
        })
    elif neural_state.arousal < 0.3:
        objects.append({
            "name": "Still Pond",
            "description": "A perfectly calm pond reflecting the sky above",
            "symbolism": "Calmness, reflection, inner peace"
        })
    
    # Motif-based objects
    for motif in neural_state.motif_tags[:2]:
        if "nature" in motif.lower():
            objects.append({
                "name": "Ancient Tree",
                "description": "A massive tree with roots extending deep into the earth",
                "symbolism": "Growth, grounding, natural wisdom"
            })
        elif "water" in motif.lower():
            objects.append({
                "name": "Crystal Stream",
                "description": "A flowing stream of pure, clear water",
                "symbolism": "Flow, cleansing, emotional release"
            })
    
    return objects

def get_mood_description(neural_state: NeuralState) -> str:
    """Generate a mood description based on neural state"""
    
    # Determine primary mood quadrant
    if neural_state.valence > 0 and neural_state.arousal > 0.5:
        return "Excited and joyful"
    elif neural_state.valence > 0 and neural_state.arousal <= 0.5:
        return "Calm and content"
    elif neural_state.valence <= 0 and neural_state.arousal > 0.5:
        return "Anxious and agitated"
    else:
        return "Sad and withdrawn"

@app.post("/generate", response_model=NarrativeResponse)
async def generate_narrative(request: NarrativeRequest):
    """Generate narrative content based on neural state"""
    try:
        neural_state = request.neural_state
        
        # Generate components
        ambient_text = generate_ambient_narration(neural_state)
        symbolic_objects = generate_symbolic_objects(neural_state)
        mood_description = get_mood_description(neural_state)
        
        # Calculate confidence based on signal strength
        signal_strength = abs(neural_state.valence) + neural_state.arousal + abs(neural_state.dominance)
        confidence = min(signal_strength / 3.0, 1.0)
        
        response = NarrativeResponse(
            ambient_text=ambient_text,
            symbolic_objects=symbolic_objects,
            mood_description=mood_description,
            confidence=confidence
        )
        
        # Cache the result
        cache_key = f"narrative:{hash(str(neural_state.dict()))}"
        redis_client.setex(cache_key, 300, json.dumps(response.dict()))
        
        logger.info(f"Generated narrative for mood: {mood_description}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating narrative: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "narrative-layer"}

@app.get("/metrics")
async def get_metrics():
    """Basic metrics endpoint"""
    return {
        "requests_processed": redis_client.get("narrative_requests") or 0,
        "cache_hits": redis_client.get("narrative_cache_hits") or 0,
        "service_uptime": "active"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)
