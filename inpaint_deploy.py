import base64
import io
import torch
from PIL import Image
import numpy as np
import modal
from pydantic import BaseModel

# Base image with common dependencies
base_image = (
    modal.Image.from_registry("pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime")
    .env({"DEBIAN_FRONTEND": "noninteractive", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"})
    .apt_install("git", "curl", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.1.2",
        "diffusers==0.34.0",
        "transformers==4.53.1",
        "accelerate",
        "xformers==0.0.23.post1",
        "fastapi",
        "pydantic",
        "Pillow",
        "numpy",
        extra_index_url="https://download.pytorch.org/whl/cu118",
    )
)

# Model weights volume
model_vol = modal.Volume.from_name("inpaint-models-vol", create_if_missing=True)
MOUNT = "/vol/models"

app = modal.App("inpaint-api", image=base_image)


# Request/Response schemas
class InpaintRequest(BaseModel):
    id: str
    image_data: str  # base64 encoded image
    mask_tl_x: int
    mask_tl_y: int
    mask_br_x: int
    mask_br_y: int
    caption: str


class InpaintResponse(BaseModel):
    id: str
    image_data: str  # base64 encoded result


# Utility functions
def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image.convert("RGB")


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def create_mask_from_coordinates(
    width: int, height: int, tl_x: int, tl_y: int, br_x: int, br_y: int
) -> Image.Image:
    """Create mask image from coordinates (white = inpaint, black = keep)"""
    mask = Image.new("L", (width, height), 0)  # Black background
    mask_array = np.array(mask)

    # Ensure coordinates are within bounds
    tl_x = max(0, min(tl_x, width))
    tl_y = max(0, min(tl_y, height))
    br_x = max(0, min(br_x, width))
    br_y = max(0, min(br_y, height))

    # Create white rectangle for inpainting area
    mask_array[tl_y:br_y, tl_x:br_x] = 255

    return Image.fromarray(mask_array, mode="L")


# Global variable for model caching
sd2_pipe = None


def get_sd2_pipe():
    """Load SD2 inpainting pipeline"""
    global sd2_pipe
    if sd2_pipe is None:
        from diffusers import StableDiffusionInpaintPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        sd2_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=dtype,
            cache_dir=f"{MOUNT}/sd2-inpaint",
        ).to(device)

        model_vol.commit()
    return sd2_pipe


# SD2 Inpainting Endpoint
@app.function(
    gpu="A100",
    timeout=300,
    volumes={MOUNT: model_vol},
)
@modal.fastapi_endpoint(method="POST", label="inpaint")
def inpaint(request: InpaintRequest):
    """Stable Diffusion 2 inpainting endpoint"""
    try:
        # Load image and create mask
        image = base64_to_image(request.image_data)
        image = image.resize((512, 512), Image.LANCZOS)

        mask = create_mask_from_coordinates(
            512,
            512,
            request.mask_tl_x,
            request.mask_tl_y,
            request.mask_br_x,
            request.mask_br_y,
        )

        # Get pipeline
        pipe = get_sd2_pipe()

        # Generate
        with torch.inference_mode():
            result = pipe(
                prompt=request.caption,
                image=image,
                mask_image=mask,
                num_inference_steps=20,
                guidance_scale=7.5,
            ).images[0]

        # Convert to base64
        result_base64 = image_to_base64(result)

        return InpaintResponse(id=request.id, image_data=result_base64)

    except Exception as e:
        print(f"SD2 inpainting error: {str(e)}")
        # Return original image on error
        return InpaintResponse(id=request.id, image_data=request.image_data)
