# NutriLens Food Segmentation API

This optional API adds **multi-instance ingredient segmentation** to the static NutriLens website. It uses the public `arunapb/yolo11l-food-segmentation` YOLO11l-seg checkpoint trained on FoodSeg103 (103 ingredient classes).

## Deploy it free on Hugging Face Spaces

1. Create a new public Space at https://huggingface.co/new-space.
2. Choose **Docker** as the Space SDK and the free CPU hardware tier.
3. Upload the contents of this `backend/` folder to the Space repository root: `app.py`, `requirements.txt`, and `Dockerfile`.
4. Wait for the first build. On first startup, the service downloads the approximately 223 MB `best.pt` checkpoint from Hugging Face; this can take a while.
5. Open the Space URL, for example `https://YOUR-SPACE-NAME.hf.space/health`. A healthy response should include `"status":"ok"`.
6. Open NutriLens, expand **Connect multi-food segmentation engine**, paste the Space base URL (without `/segment`), and press **Save**.
7. Upload or capture a food photo. NutriLens will call `POST /segment` and display each returned ingredient mask and label.

## API

- `GET /` service information
- `GET /health` health status
- `POST /segment` multipart form field `file` (JPG, PNG or WEBP, maximum 10 MB)

The response contains image dimensions and an `instances` array. Each instance has a class label, model score, bounding box and polygon contour in source-image pixel coordinates.

## Model coverage and limitations

FoodSeg103 has 103 ingredient classes, including common produce, rice, corn, noodles, meat, fish, tofu, mushrooms and other ingredients. It is **not a complete India/Nepal dish catalogue**. Foods such as momo, dal bhat, thukpa, idli, dosa, biryani and regional curries may need separate dish recognition; NutriLens keeps its CLIP-based photo suggestions and expanded South Asian food candidates for that purpose. Segmentation may miss tiny, mixed, covered or overlapping ingredients, and a region's label can be wrong. Confirm results before using nutrition estimates.

The backend currently runs inference on CPU. First startup and inference can be slow on free CPU hardware; it is not a real-time video segmentation service. Camera capture is supported, but the captured still image is segmented after capture.

## Privacy and cost

The browser sends the selected image to the Space URL you configure. Do not connect a public backend if you need strictly on-device processing. Free hardware availability, sleeping/cold-start behaviour and usage limits are controlled by Hugging Face and may change. Public Spaces are internet-accessible; do not upload sensitive images.

## Model and licensing

Model: https://huggingface.co/arunapb/yolo11l-food-segmentation (Apache-2.0 model card; FoodSeg103 benchmark). The API code in this folder is provided under the MIT License. Review upstream model and dataset terms before commercial redistribution.
