from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from API.inference import InferenceArtifacts, load_artifacts, predict_image_bytes
from API.schemas import PredictionResponse
from API.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once on startup
    try:
        app.state.artifacts = load_artifacts(
            data_dir=settings.data_dir,
            model_file=settings.model_file,
            model_name=settings.model_name,
            device=settings.device,
        )
    except Exception as e:
        # Fail fast at startup if the model can't load
        raise RuntimeError(f"Failed to load inference artifacts: {e}") from e

    yield
    # no teardown needed


app = FastAPI(title="Plant Classifier API", version="1.0.0", lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    # Basic content-type check (not perfect, but helpful)
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    artifacts: InferenceArtifacts = app.state.artifacts

    try:
        class_name, confidence = predict_image_bytes(artifacts, image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        # avoid leaking internals
        raise HTTPException(status_code=500, detail="Inference failed.") from e

    return PredictionResponse(class_name=class_name, confidence=confidence)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
