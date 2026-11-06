# NutriLens · Render Free segmentation backend

This optional API adds multi-food ingredient segmentation to the static NutriLens website. It uses the YOLO11l-seg checkpoint `arunapb/yolo11l-food-segmentation`, trained on FoodSeg103 (103 ingredient classes).

## Deploy to Render without Docker

1. Push this repository to GitHub and sign in to [Render](https://render.com/).
2. Choose **New → Web Service** and connect `Backtakg/NutriLens`.
3. Set **Root Directory** to `backend`.
4. Choose the **Python** runtime.
5. Set **Build Command** to `pip install -r requirements.txt`.
6. Set **Start Command** to `uvicorn app:app --host 0.0.0.0 --port $PORT`.
7. Select the **Free** instance type and create the service.
8. After deployment, open `https://YOUR-SERVICE.onrender.com/health`. It should return JSON with `"status":"ok"`.
9. Copy the service's base HTTPS URL. In NutriLens, open **Connect multi-food segmentation engine**, paste the URL, save, and upload a test image.

No Dockerfile, paid hardware, or paid subscription is required for this setup. Render builds the Python service from the requirements file.

## Important free-tier performance warning

The chosen YOLO11l checkpoint is approximately 223 MB before Python, PyTorch, Ultralytics and inference memory are counted. Render Free has a small memory allowance and CPU-only execution. The service may fail to load the model, run out of memory, or take a long time per image. This deployment deliberately keeps YOLO11l as the first accuracy-focused experiment; it does not promise that the model will fit or perform acceptably on every free instance.

The model is loaded lazily on the first segmentation request so the health endpoint can respond before the model is loaded. The first request can take substantially longer because it downloads the checkpoint and initializes the model. Render Free services also sleep when idle and need time to wake.

If Render reports an out-of-memory restart, do not repeatedly send large images. Review the service logs first. We can then measure whether further CPU/memory optimization is realistic before considering any model change. No paid upgrade is assumed.

## API

- `GET /` — service information.
- `GET /health` — health status and whether the model has been loaded.
- `POST /segment` — multipart form upload with the image in the `file` field.

The segmentation response contains the source image width and height plus an `instances` array. Each instance includes a label, confidence score, bounding box and polygon in original-image pixel coordinates. The API accepts image uploads up to 12 MB and resizes very large images for inference while mapping masks back to the original dimensions.

## Model coverage and limitations

FoodSeg103 covers 103 ingredient categories; it is not a complete catalogue of Nepali and Indian dishes. It does not guarantee recognition of momo, dal-bhat, thukpa, dosa, biryani, curries or every regional preparation as complete dishes. Dish-level recognition is a separate task. Small, hidden, overlapping or mixed ingredients can be missed or mislabeled. Validate outputs against a representative Nepal/India test set before making accuracy claims.

Nutrition calculations remain in the NutriLens website and are approximate per-100 g reference estimates. A segmentation mask does not determine serving weight, recipe, hidden ingredients or cooking oil.

## Privacy and public access

When a user configures this API URL, NutriLens sends the selected image to this Render service for processing. The endpoint is publicly reachable and this MVP does not include authentication, rate limiting or per-user quotas. Do not upload sensitive images. Add appropriate access controls before exposing it to unrestricted public traffic.

## License

The upstream model card states Apache-2.0. Review upstream model and dataset terms before commercial redistribution. Backend source code is covered by the repository's MIT License.
