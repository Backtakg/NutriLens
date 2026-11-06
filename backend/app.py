import math
from functools import lru_cache

import spaces  # Must be imported before torch/ultralytics on ZeroGPU.
import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from ultralytics import YOLO

MODEL_REPO = "arunapb/yolo11l-food-segmentation"
MODEL_FILE = "best.pt"
CONFIDENCE = 0.22
IMAGE_SIZE = 640
MAX_DETECTIONS = 30
MAX_SIDE = 4096

# ZeroGPU requires models to be loaded at module scope and moved to "cuda".
# The spaces package safely virtualizes this placement until a GPU is allocated.
weights = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
model = YOLO(weights)
model.to("cuda")


def segment_image(image: Image.Image):
    """Validate and segment a single uploaded image."""
    if image is None:
        raise gr.Error("Please upload a food photo first.")

    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width < 32 or height < 32:
        raise gr.Error("Image is too small. Please use an image at least 32 × 32 pixels.")
    if max(width, height) > MAX_SIDE:
        scale = MAX_SIDE / max(width, height)
        image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
        width, height = image.size

    result = model.predict(
        source=image,
        conf=CONFIDENCE,
        iou=0.50,
        imgsz=IMAGE_SIZE,
        max_det=MAX_DETECTIONS,
        retina_masks=True,
        device="cuda",
        verbose=False,
    )[0]

    instances = []
    if result.masks is not None and result.boxes is not None:
        for index, box in enumerate(result.boxes):
            if index >= len(result.masks.xy):
                break
            points = result.masks.xy[index]
            if points is None or len(points) < 3:
                continue
            stride = max(1, math.ceil(len(points) / 240))
            polygon = [[round(float(x), 1), round(float(y), 1)] for x, y in points[::stride]]
            if len(polygon) < 3:
                continue
            class_id = int(box.cls[0].item())
            label = result.names.get(class_id, str(class_id))
            instances.append({
                "label": str(label),
                "confidence": round(float(box.conf[0].item()), 4),
                "bbox": [round(float(v), 1) for v in box.xyxy[0].tolist()],
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


@spaces.GPU(duration=45)
def segment(image: Image.Image):
    return segment_image(image)


with gr.Blocks(title="NutriLens Food Segmentation") as demo:
    gr.Markdown(
        "# NutriLens · Multi-food segmentation\n"
        "Upload a meal photo to detect ingredient regions with FoodSeg103. "
        "This model covers 103 ingredient classes; it is not a complete Nepali or Indian dish classifier."
    )
    image_input = gr.Image(type="pil", label="Food photo")
    run_button = gr.Button("Segment food", variant="primary")
    output = gr.JSON(label="Segmentation result")
    run_button.click(fn=segment, inputs=image_input, outputs=output, api_name="segment")
    gr.Markdown(
        "Images uploaded here are processed by this public Space. "
        "Results can miss small, hidden, overlapping or mixed ingredients. "
        "Nutrition estimates are handled by the NutriLens website, not this model."
    )

demo.queue(max_size=8).launch()
