"""
DreamWalk Texture Generator Service

Generates procedural textures and skyboxes using Stable Diffusion for world generation.
"""

import asyncio
import logging
import os
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as redis
import httpx
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, ControlNetModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response

from .models.texture_models import (
    TextureRequest, TextureResponse, TextureGenerationConfig,
    BiomeConfig, StylePreset
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
TEXTURES_GENERATED = Counter(
    'textures_generated_total',
    'Total number of textures generated',
    ['texture_type', 'biome_type']
)

TEXTURE_GENERATION_DURATION = Histogram(
    'texture_generation_duration_seconds',
    'Time spent generating textures',
    ['texture_type', 'model_type']
)

ACTIVE_GENERATIONS = Gauge(
    'active_texture_generations_total',
    'Number of active texture generations'
)

MODEL_INFERENCE_TIME = Histogram(
    'model_inference_duration_seconds',
    'Model inference time for texture generation',
    ['model_name', 'resolution']
)

app = FastAPI(
    title="DreamWalk Texture Generator",
    description="Procedural texture and skybox generation service",
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
sd_pipeline = None
controlnet = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Biome configurations
BIOME_CONFIGS = {
    "lush_forest": {
        "base_prompt": "lush green forest, tall trees, sunlight filtering through leaves, magical atmosphere",
        "color_palette": [0.2, 0.8, 0.3],
        "style": "nature, fantasy, vibrant"
    },
    "peaceful_garden": {
        "base_prompt": "peaceful zen garden, cherry blossoms, stone pathways, tranquil water features",
        "color_palette": [0.6, 0.8, 0.4],
        "style": "zen, minimalist, calming"
    },
    "stormy_mountains": {
        "base_prompt": "dramatic mountain peaks, storm clouds, lightning, rocky terrain, epic scale",
        "color_palette": [0.3, 0.4, 0.6],
        "style": "dramatic, epic, stormy"
    },
    "desert_wasteland": {
        "base_prompt": "vast desert wasteland, sand dunes, harsh sunlight, barren landscape, post-apocalyptic",
        "color_palette": [0.9, 0.7, 0.3],
        "style": "desert, harsh, barren"
    },
    "neutral_plains": {
        "base_prompt": "rolling plains, gentle hills, clear sky, peaceful countryside",
        "color_palette": [0.5, 0.6, 0.4],
        "style": "neutral, peaceful, natural"
    },
    "zen_garden": {
        "base_prompt": "zen meditation garden, raked sand, stones, bamboo, minimalist design",
        "color_palette": [0.7, 0.6, 0.5],
        "style": "zen, meditation, minimalist"
    },
    "chaotic_void": {
        "base_prompt": "chaotic void, swirling energy, dark matter, cosmic storm, abstract",
        "color_palette": [0.1, 0.1, 0.2],
        "style": "abstract, chaotic, dark"
    },
    "crystal_caverns": {
        "base_prompt": "crystal caverns, glowing gems, underground caves, magical crystals, ethereal light",
        "color_palette": [0.3, 0.6, 0.9],
        "style": "crystal, magical, underground"
    },
    "surreal_landscape": {
        "base_prompt": "surreal dreamscape, impossible geometry, floating islands, dreamlike atmosphere",
        "color_palette": [0.8, 0.4, 0.9],
        "style": "surreal, dreamlike, impossible"
    }
}


@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global redis_client, sd_pipeline, controlnet
    
    logger.info("Starting texture generator service...")
    
    # Initialize Redis connection
    redis_client = redis.from_url("redis://redis:6379", decode_responses=True)
    await redis_client.ping()
    
    # Initialize Stable Diffusion pipeline
    try:
        logger.info("Loading Stable Diffusion model...", device=device)
        
        # Load base Stable Diffusion pipeline
        sd_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        sd_pipeline = sd_pipeline.to(device)
        
        # Enable memory efficient attention
        sd_pipeline.enable_attention_slicing()
        sd_pipeline.enable_model_cpu_offload()
        
        logger.info("Stable Diffusion model loaded successfully")
        
    except Exception as e:
        logger.error("Failed to load Stable Diffusion model", error=str(e))
        # Continue without model for now
    
    # Create output directory
    os.makedirs("/app/output", exist_ok=True)
    
    logger.info("Texture generator service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if redis_client:
        await redis_client.close()
    logger.info("Texture generator service stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": {
            "stable_diffusion": sd_pipeline is not None,
            "controlnet": controlnet is not None
        },
        "device": device
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")


@app.post("/generate", response_model=TextureResponse)
async def generate_textures(request: TextureRequest, background_tasks: BackgroundTasks):
    """Generate textures for world state"""
    try:
        generation_id = str(uuid.uuid4())
        
        # Validate request
        if not request.texture_types:
            raise HTTPException(status_code=400, detail="No texture types specified")
        
        # Start generation in background
        background_tasks.add_task(
            _generate_textures_background,
            generation_id,
            request
        )
        
        return TextureResponse(
            generation_id=generation_id,
            status="started",
            message="Texture generation started",
            texture_urls=[]
        )
        
    except Exception as e:
        logger.error("Texture generation request failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate/{generation_id}/status")
async def get_generation_status(generation_id: str):
    """Get status of texture generation"""
    try:
        status_data = await redis_client.get(f"texture_gen:{generation_id}")
        if not status_data:
            raise HTTPException(status_code=404, detail="Generation not found")
        
        status = json.loads(status_data)
        return status
        
    except Exception as e:
        logger.error("Failed to get generation status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/presets/biomes")
async def get_biome_presets():
    """Get available biome presets"""
    return {
        "biomes": list(BIOME_CONFIGS.keys()),
        "configs": BIOME_CONFIGS
    }


@app.post("/presets/biomes/{biome_type}/customize")
async def customize_biome_preset(biome_type: str, customization: Dict[str, Any]):
    """Customize a biome preset"""
    try:
        if biome_type not in BIOME_CONFIGS:
            raise HTTPException(status_code=404, detail="Biome not found")
        
        # Create custom biome config
        custom_config = BIOME_CONFIGS[biome_type].copy()
        custom_config.update(customization)
        
        # Generate textures for custom biome
        generation_id = str(uuid.uuid4())
        
        # Store custom config
        await redis_client.setex(
            f"custom_biome:{generation_id}",
            3600,  # 1 hour TTL
            json.dumps(custom_config)
        )
        
        return {
            "generation_id": generation_id,
            "custom_config": custom_config
        }
        
    except Exception as e:
        logger.error("Failed to customize biome preset", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_textures_background(generation_id: str, request: TextureRequest):
    """Generate textures in background"""
    try:
        ACTIVE_GENERATIONS.inc()
        
        # Update status
        await _update_generation_status(generation_id, "generating", "Starting generation...")
        
        generated_textures = []
        
        for texture_type in request.texture_types:
            with TEXTURE_GENERATION_DURATION.labels(
                texture_type=texture_type,
                model_type="stable_diffusion"
            ).time():
                
                # Generate texture
                texture_url = await _generate_single_texture(
                    texture_type,
                    request.world_state,
                    generation_id
                )
                
                if texture_url:
                    generated_textures.append(texture_url)
                    
                    TEXTURES_GENERATED.labels(
                        texture_type=texture_type,
                        biome_type=request.world_state.get("biome_type", "unknown")
                    ).inc()
        
        # Update final status
        await _update_generation_status(
            generation_id,
            "completed",
            "Generation completed",
            generated_textures
        )
        
        logger.info("Texture generation completed", generation_id=generation_id, count=len(generated_textures))
        
    except Exception as e:
        logger.error("Background texture generation failed", error=str(e), generation_id=generation_id)
        await _update_generation_status(generation_id, "failed", f"Generation failed: {str(e)}")
        
    finally:
        ACTIVE_GENERATIONS.dec()


async def _generate_single_texture(texture_type: str, world_state: Dict[str, Any], generation_id: str) -> Optional[str]:
    """Generate a single texture"""
    try:
        biome_type = world_state.get("biome_type", "neutral_plains")
        biome_config = BIOME_CONFIGS.get(biome_type, BIOME_CONFIGS["neutral_plains"])
        
        # Create prompt based on texture type and biome
        prompt = _create_texture_prompt(texture_type, biome_config, world_state)
        
        # Generate image
        if sd_pipeline is not None:
            with MODEL_INFERENCE_TIME.labels(model_name="stable_diffusion", resolution="512x512").time():
                
                # Set generation parameters
                num_inference_steps = 20
                guidance_scale = 7.5
                
                # Generate image
                image = sd_pipeline(
                    prompt=prompt,
                    negative_prompt="blurry, low quality, distorted, ugly",
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=512,
                    height=512
                ).images[0]
                
        else:
            # Fallback: generate simple colored image
            image = _generate_fallback_texture(texture_type, biome_config)
        
        # Save image
        filename = f"{generation_id}_{texture_type}.png"
        filepath = os.path.join("/app/output", filename)
        image.save(filepath)
        
        # Return relative URL
        return f"textures/{filename}"
        
    except Exception as e:
        logger.error("Single texture generation failed", error=str(e), texture_type=texture_type)
        return None


def _create_texture_prompt(texture_type: str, biome_config: Dict[str, Any], world_state: Dict[str, Any]) -> str:
    """Create prompt for texture generation"""
    base_prompt = biome_config["base_prompt"]
    style = biome_config["style"]
    
    # Texture type specific modifications
    if texture_type == "skybox":
        prompt = f"panoramic {base_prompt}, 360-degree view, seamless, sky, clouds, atmospheric"
    elif texture_type == "terrain":
        prompt = f"ground texture, {base_prompt}, detailed surface, seamless tileable, natural"
    elif texture_type == "ambient":
        prompt = f"atmospheric overlay, {base_prompt}, subtle effects, ambient lighting"
    else:
        prompt = base_prompt
    
    # Add style
    prompt += f", {style} style, high quality, detailed"
    
    # Add weather effects
    weather_intensity = world_state.get("weather_intensity", 0.5)
    if weather_intensity > 0.7:
        prompt += ", stormy weather, dramatic lighting"
    elif weather_intensity < 0.3:
        prompt += ", calm weather, peaceful atmosphere"
    
    # Add color palette influence
    color_palette = world_state.get("color_palette", [0.5, 0.5, 0.5])
    if color_palette:
        # Convert RGB to descriptive terms
        r, g, b = color_palette[:3]
        if r > 0.7:
            prompt += ", warm tones, reddish"
        if g > 0.7:
            prompt += ", green tones, natural"
        if b > 0.7:
            prompt += ", blue tones, cool"
    
    return prompt


def _generate_fallback_texture(texture_type: str, biome_config: Dict[str, Any]) -> Image.Image:
    """Generate fallback texture when model is not available"""
    try:
        # Get biome color palette
        color_palette = biome_config.get("color_palette", [0.5, 0.5, 0.5])
        base_color = tuple(int(c * 255) for c in color_palette[:3])
        
        # Create simple gradient or pattern based on texture type
        if texture_type == "skybox":
            # Create sky gradient
            img = Image.new('RGB', (512, 512))
            for y in range(512):
                ratio = y / 512
                # Gradient from light blue to darker blue
                color = (
                    int(base_color[0] * (1 - ratio * 0.3)),
                    int(base_color[1] * (1 - ratio * 0.2)),
                    int(base_color[2] * (1 - ratio * 0.1))
                )
                for x in range(512):
                    img.putpixel((x, y), color)
        
        elif texture_type == "terrain":
            # Create terrain pattern
            img = Image.new('RGB', (512, 512))
            for y in range(512):
                for x in range(512):
                    # Simple noise pattern
                    noise = int(50 * np.sin(x * 0.1) * np.cos(y * 0.1))
                    color = (
                        max(0, min(255, base_color[0] + noise)),
                        max(0, min(255, base_color[1] + noise)),
                        max(0, min(255, base_color[2] + noise))
                    )
                    img.putpixel((x, y), color)
        
        else:
            # Default solid color
            img = Image.new('RGB', (512, 512), base_color)
        
        return img
        
    except Exception as e:
        logger.error("Fallback texture generation failed", error=str(e))
        # Return simple white image as last resort
        return Image.new('RGB', (512, 512), (255, 255, 255))


async def _update_generation_status(generation_id: str, status: str, message: str, texture_urls: List[str] = None):
    """Update generation status in Redis"""
    try:
        status_data = {
            "generation_id": generation_id,
            "status": status,
            "message": message,
            "texture_urls": texture_urls or [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await redis_client.setex(
            f"texture_gen:{generation_id}",
            3600,  # 1 hour TTL
            json.dumps(status_data)
        )
        
    except Exception as e:
        logger.error("Failed to update generation status", error=str(e))


if __name__ == "__main__":
    import uvicorn
    import json
    uvicorn.run(app, host="0.0.0.0", port=8005)
