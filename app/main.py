import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.generator import SceneBackgroundGenerator

app = FastAPI(title="Scene BG AI", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

generator = SceneBackgroundGenerator()


class GenerateRequest(BaseModel):
    prompt: str
    style: str = "cinematic"
    mood: str = "epic"
    quality: int = 7
    detail: int = 7
    saturation: int = 7


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/generate")
def generate_background(payload: GenerateRequest):
    try:
        image_data, status = generator.generate(
            prompt=payload.prompt,
            style=payload.style,
            mood=payload.mood,
            quality=max(1, min(10, payload.quality)),
            detail=max(1, min(10, payload.detail)),
            saturation=max(1, min(10, payload.saturation)),
        )
        return {"image": image_data, "status": status}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(exc)}") from exc
