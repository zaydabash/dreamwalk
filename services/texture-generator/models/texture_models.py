"""
Texture generation data models for DreamWalk
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class TextureRequest(BaseModel):
    """Request to generate textures for a world state"""

    session_id: str = Field(default="", description="Session identifier")
    world_state: Dict[str, Any] = Field(default_factory=dict, description="World state to texture")
    texture_types: List[str] = Field(default_factory=list, description="Texture types to generate")

    @validator("texture_types")
    def validate_texture_types(cls, v):
        valid_types = ["skybox", "terrain", "ambient"]
        for texture_type in v:
            if texture_type not in valid_types:
                raise ValueError(
                    f"Invalid texture type: {texture_type}. Must be one of {valid_types}"
                )
        return v


class TextureResponse(BaseModel):
    """Response from a texture generation request"""

    generation_id: str = Field(..., description="Unique generation identifier")
    status: str = Field(..., description="Generation status")
    message: str = Field(default="", description="Status message")
    texture_urls: List[str] = Field(default_factory=list, description="URLs of generated textures")


class TextureGenerationConfig(BaseModel):
    """Configuration for the Stable Diffusion texture pipeline"""

    model_name: str = Field(
        default="runwayml/stable-diffusion-v1-5", description="Diffusion model name"
    )
    num_inference_steps: int = Field(default=20, ge=1, description="Number of denoising steps")
    guidance_scale: float = Field(default=7.5, ge=0.0, description="Classifier-free guidance scale")
    width: int = Field(default=512, description="Output image width")
    height: int = Field(default=512, description="Output image height")
    negative_prompt: str = Field(
        default="blurry, low quality, distorted, ugly",
        description="Negative prompt applied to generations",
    )


class BiomeConfig(BaseModel):
    """Configuration for a single biome preset"""

    base_prompt: str = Field(..., description="Base text-to-image prompt for this biome")
    color_palette: List[float] = Field(..., description="RGB color palette (normalized)")
    style: str = Field(..., description="Style descriptors for this biome")

    @validator("color_palette")
    def validate_color_palette(cls, v):
        if len(v) != 3:
            raise ValueError("Color palette must have exactly 3 RGB values")
        for color in v:
            if not 0.0 <= color <= 1.0:
                raise ValueError("Color values must be between 0.0 and 1.0")
        return v


class StylePreset(BaseModel):
    """Reusable style modifier for texture prompts"""

    name: str = Field(..., description="Preset name")
    prompt_suffix: str = Field(default="", description="Text appended to generation prompts")
    negative_prompt: Optional[str] = Field(
        default=None, description="Override negative prompt for this preset"
    )
