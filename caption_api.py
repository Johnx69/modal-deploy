import modal
import base64
import io
from PIL import Image
from pydantic import BaseModel
import json


# Define the Modal app for captioning
app = modal.App("image-captioning-api")


# Lightweight base image for API-only captioning (no GPU needed)
base_image = (
    modal.Image.from_registry("python:3.11-slim")
    .env(
        {
            "DEBIAN_FRONTEND": "noninteractive",
        }
    )
    .apt_install("curl", "libgl1", "libglib2.0-0")
    .pip_install(
        "fastapi",
        "pydantic",
        "pillow",
        "numpy",
        "openai",
        "google-generativeai",
    )
)


# Request/Response models
class CaptionRequest(BaseModel):
    id: str
    image_data: str  # base64 encoded image


class CaptionResponse(BaseModel):
    id: str
    caption: str


# Shared utility functions
def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]

    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes))

    # Ensure image is 512x512 and RGB
    if image.size != (512, 512):
        image = image.resize((512, 512), Image.LANCZOS)
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


# =============================================================================
# CAPTIONING ENDPOINTS
# =============================================================================


@app.function(
    secrets=[modal.Secret.from_name("openai-secret")],
    image=base_image,
)
@modal.fastapi_endpoint(method="POST", label="caption-gpt4")
def caption_with_gpt4(request: CaptionRequest):
    """Generate caption using GPT-4o"""
    import os
    from openai import OpenAI

    PROMPT = """
    Task: Describe the visible detail content of this image in one concise sentence (60 words).
    Guidelines:
        • Begin with a concrete noun (“A snowy mountain peak towers…”).
        • Mention key objects, actions, colors, spatial layout, and setting.
        • Use specific nouns (“golden retriever,” “red double‑decker bus”) and simple, neutral language—no brand names, emotions, camera terms, or style words.
        • Avoid redundant adjectives or adverbs.

    Output: Return the sentence only. Nothing else—no IDs, JSON, or commentary.
    """

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    try:
        # Prepare image for OpenAI
        image_data_url = f"data:image/png;base64,{request.image_data}"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT,
                        },
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            max_tokens=300,
        )

        caption = response.choices[0].message.content
        return CaptionResponse(id=request.id, caption=caption)

    except Exception as e:
        return CaptionResponse(id=request.id, caption=f"Error: {str(e)}")


@app.function(
    secrets=[modal.Secret.from_name("gemini-secret")],
    image=base_image,
)
@modal.fastapi_endpoint(method="POST", label="caption-gemini")
def caption_with_gemini(request: CaptionRequest):
    """Generate caption using Gemini 2.5 Flash"""
    import os
    import google.generativeai as genai

    PROMPT = """
    Task: Describe the visible content of this image in one concise sentence (maximum 60 words).
    Guidelines:
        • Begin with a concrete noun (“A snowy mountain peak towers…”).
        • Mention key objects, actions, colors, spatial layout, and setting.
        • Use specific nouns (“golden retriever,” “red double‑decker bus”) and simple, neutral language—no brand names, emotions, camera terms, or style words.
        • Avoid redundant adjectives or adverbs.

    Output: Return the sentence only. Nothing else—no IDs, JSON, or commentary.
    """
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    try:
        # Decode image
        image = decode_base64_image(request.image_data)

        # Use Gemini 2.5 Flash for captioning
        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content(
            [
                PROMPT,
                image,
            ]
        )

        caption = response.text
        return CaptionResponse(id=request.id, caption=caption)

    except Exception as e:
        return CaptionResponse(id=request.id, caption=f"Error: {str(e)}")


# =============================================================================
# HEALTH CHECK ENDPOINT
# =============================================================================


@app.function(image=base_image)
@modal.fastapi_endpoint(method="GET", label="health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Image captioning API is running"}


if __name__ == "__main__":
    print("Image Captioning API ready for deployment!")
    print("\nCAPTIONING ENDPOINTS:")
    print("- /caption-gpt4 (POST) - Caption with GPT-4o")
    print("- /caption-gemini (POST) - Caption with Gemini 2.5 Flash")
    print("\nUTILITY ENDPOINTS:")
    print("- /health (GET) - Health check")
