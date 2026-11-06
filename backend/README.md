---
title: NutriLens Food Segmentation
sdk: gradio
app_file: app.py
python_version: "3.12"
suggested_hardware: zero-a10g
---

# NutriLens · Free ZeroGPU segmentation backend

This is the optional multi-food segmentation service for the static NutriLens website. It uses the public `arunapb/yolo11l-food-segmentation` YOLO11l-seg checkpoint trained on FoodSeg103 (103 ingredient classes).

## Deploy without Docker

1. Sign in to Hugging Face and open https://huggingface.co/new-space.
2. Choose a **Gradio** Space (not Docker), set it to **Public**, and create it.
3. In the Space Settings → Hardware, select **ZeroGPU** (shown as ZeroGPU / zero-a10g).
4. Upload the contents of this folder to the Space repository root: `app.py`, `requirements.txt`, and this `README.md`.
5. Wait for the build and initial model download. The model checkpoint is approximately 223 MB.
6. Open the Space page and check that the Gradio interface loads.
7. In NutriLens, expand **Connect multi-food segmentation engine** and paste the Space URL, for example `https://YOUR-USERNAME-nutrilens-food-api.hf.space`. Save, then upload a test photo.

The website connects using the official Gradio JavaScript client and calls the `/segment` endpoint. No Hugging Face token is required for a public Space.

## Free-tier limits

Hugging Face's current documentation says eligible free personal accounts in good standing (verified email and account older than 30 days) can host up to two ZeroGPU Spaces. Free accounts have a limited shared GPU allowance (currently documented as 5 GPU minutes per 24-hour usage window). Availability, quota and eligibility are controlled by Hugging Face and may change. If ZeroGPU is unavailable for your account, do not select paid hardware; keep the app disconnected until an eligible free option is available.

## Model coverage and limitations

FoodSeg103 has 103 ingredient classes, including common produce, rice, corn, noodles, meat, fish, tofu and mushrooms. It is **not a complete India/Nepal dish catalogue**. Dishes such as momo, dal bhat, thukpa, idli, dosa, biryani and regional curries need separate dish-level recognition and/or region-specific training. Small, mixed, hidden or overlapping ingredients can be missed or mislabeled. Confirm results before using nutrition estimates.

The endpoint returns source-image dimensions and an `instances` array. Each instance includes a label, confidence score, bounding box and polygon contour in source-image pixel coordinates.

## Privacy and public access

The browser sends the selected image to the public Space URL configured in NutriLens. Do not connect it if you need strictly on-device processing or are uploading sensitive images. Public Spaces can be called by other people and may consume the Space owner's limited GPU allowance. This MVP has no user authentication or per-user quota controls; keep an eye on usage and disable the connection when not testing.

## Model and licensing

Model: https://huggingface.co/arunapb/yolo11l-food-segmentation (model card states Apache-2.0). The backend code is provided under the repository's MIT License. Review upstream model and dataset terms before commercial redistribution.
