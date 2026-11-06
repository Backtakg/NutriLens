import io
import logging
import math
import os
from functools import lru_cache

# Keep CPU thread usage modest on small free instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nutrilens")

MODEL_REPO = "arunapb/yolo11l-food-segmentation"
MODEL_FILE = "best.pt"
CONFIDENCE = 0.22
IMAGE_SIZE = 640
MAX_DETECTIONS = 20
MAX_SIDE = 2048
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

app = FastAPI(title="NutriLens Food Segmentation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_model():
    """Download and load the YOLO checkpoint only when the first image is analyzed."""
    import torch

    torch.set_num_threads(1)
    weights = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    model = YOLO(weights)
    model.to("cpu")
    return model


@app.get("/")
def root():
    return {"service": "NutriLens Food Segmentation API", "health": "/health"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_REPO,
        "model_loaded": load_model.cache_info().currsize > 0,
        "runtime": "cpu",
    }


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload a valid image file.")

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum upload size is 12 MB.")
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    try:
        image = Image.open(io.BytesIO(raw))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The uploaded file could not be read as an image.")

    original_width, original_height = image.size
    if original_width < 32 or original_height < 32:
        raise HTTPException(status_code=400, detail="Image is too small. Use an image at least 32 × 32 pixels.")

    scale = min(1.0, MAX_SIDE / max(original_width, original_height))
    if scale < 1.0:
        image = image.resize(
            (round(original_width * scale), round(original_height * scale)),
            Image.Resampling.LANCZOS,
        )

    try:
        model = load_model()
        result = model.predict(
            source=image,
            conf=CONFIDENCE,
            iou=0.50,
            imgsz=IMAGE_SIZE,
            max_det=MAX_DETECTIONS,
            retina_masks=False,
            device="cpu",
            verbose=False,
        )[0]
    except Exception as exc:
        logger.exception("Food segmentation inference failed")
        raise HTTPException(
            status_code=503,
            detail="The segmentation model could not run on this free CPU instance. Check Render logs for details.",
        ) from exc

    instances = []
    if result.masks is not None and result.boxes is not None:
        for index, box in enumerate(result.boxes):
            if index >= len(result.masks.xy):
                break
            points = result.masks.xy[index]
            if points is None or len(points) < 3:
                continue
            stride = max(1, math.ceil(len(points) / 240))
            polygon = [
                [round(float(x) / scale, 1), round(float(y) / scale, 1)]
                for x, y in points[::stride]
            ]
            if len(polygon) < 3:
                continue
            class_id = int(box.cls[0].item())
            label = result.names.get(class_id, str(class_id))
            bbox = [round(float(v) / scale, 1) for v in box.xyxy[0].tolist()]
            instances.append({
                "label": str(label),
                "confidence": round(float(box.conf[0].item()), 4),
                "bbox": bbox,
                "polygon": polygon,
            })

    return {
        "width": original_width,
        "height": original_height,
        "model": MODEL_REPO,
        "classes": 103,
        "instances": instances,
        "note": (
            "FoodSeg103 ingredient segmentation. It does not identify every complete regional dish; "
            "small, hidden, overlapping or mixed ingredients may be missed."
        ),
    }
