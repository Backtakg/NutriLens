import io
import os
import math
import threading
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

MAX_BYTES = 10 * 1024 * 1024
MODEL_REPO = os.getenv("MODEL_REPO", "arunapb/yolo11l-food-segmentation")
MODEL_FILE = os.getenv("MODEL_FILE", "best.pt")
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.22"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "30"))

app = FastAPI(title="NutriLens Food Segmentation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_model_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_model():
    """Download the public FoodSeg103 YOLO11 segmentation checkpoint once."""
    weights_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return YOLO(weights_path)


@app.get("/")
def root():
    return {
        "service": "NutriLens Food Segmentation API",
        "status": "ready",
        "model": MODEL_REPO,
        "endpoint": "POST /segment",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": get_model.cache_info().currsize > 0}


@app.post("/segment")
async def segment_food(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG or WEBP image.")
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    width, height = image.size
    try:
        model = get_model()
        # Ultralytics model inference is serialized to avoid concurrent CPU memory spikes.
        with _model_lock:
            result = model.predict(
                source=image,
                conf=CONFIDENCE,
                iou=0.50,
                imgsz=IMAGE_SIZE,
                max_det=MAX_DETECTIONS,
                retina_masks=True,
                device="cpu",
                verbose=False,
            )[0]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Food segmentation failed: {str(exc)[:220]}")

    instances = []
    if result.masks is not None and result.boxes is not None:
        polygons = result.masks.xy
        for index, box in enumerate(result.boxes):
            if index >= len(polygons):
                break
            points = polygons[index]
            if points is None or len(points) < 3:
                continue
            # Bound response size while retaining the contour shape.
            stride = max(1, math.ceil(len(points) / 240))
            polygon = [[round(float(x), 1), round(float(y), 1)] for x, y in points[::stride]]
            if len(polygon) < 3:
                continue
            class_id = int(box.cls[0].item())
            label = result.names.get(class_id, str(class_id))
            confidence = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            instances.append({
                "label": str(label),
                "confidence": round(confidence, 4),
                "bbox": [round(float(v), 1) for v in xyxy],
                "polygon": polygon,
            })

    return {
        "width": width,
        "height": height,
        "model": MODEL_REPO,
        "classes": 103,
        "instances": instances,
        "note": "FoodSeg103 ingredient segmentation. Not every regional dish or overlapping ingredient is guaranteed to be detected.",
    }
