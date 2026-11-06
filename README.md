# NutriLens

**See your food differently.** A privacy-first browser food studio for image cutouts, AI-assisted food suggestions, and editable nutrition estimates.

## Current experience

- Responsive, minimal food-image workspace.
- JPG, PNG, and WEBP upload with drag-and-drop.
- Client-side background removal using IMG.LY IS-Net FP16, transparent PNG output, progress feedback, and GPU-to-CPU fallback.
- Automatic background removal after upload or camera capture, with a bottom-to-top reveal animation.
- Live camera capture using the device camera (HTTPS and browser permission required).
- Zero-shot image-to-food matching using CLIP through Transformers.js.
- Expanded food vocabulary spanning ingredients, common Indian dishes, and Nepali dishes (including momo, dal bhat, sel roti, kwati, gundruk soup, thukpa, and more).
- Optional multi-food instance segmentation API using a YOLO11l-seg model trained on FoodSeg103 (103 ingredient classes), with per-region polygon masks, labels, and confidence scores.
- Ingredient nutrition mappings for supported segmentation classes; unknown classes remain visible without fabricated nutrition values.
- AI suggestions are shown for user confirmation; NutriLens does not silently turn a weak prediction into a confirmed food.
- Editable serving sizes with estimated calories, protein, carbohydrates, and fat.
- Local analysis history stored in the current browser.

## Run it

This is a static browser app; no build step or application server is required.

1. Clone the repository.
2. Serve the repository root using a local static server, for example:

   ```bash
   python -m http.server 8000
   ```

3. Open `http://localhost:8000`.

The first background-removal and AI-recognition runs need an internet connection to download browser libraries and model weights. Model assets are cached by the browser when supported. Background removal and CLIP suggestions run in the browser. Multi-food segmentation is an optional backend: images are sent to the API URL that the user explicitly configures in the food panel.

## Multi-food segmentation backend

The optional backend is in [`backend/`](backend/). It uses the [YOLO11l FoodSeg103 model](https://huggingface.co/arunapb/yolo11l-food-segmentation), a 103-class ingredient instance-segmentation checkpoint. Deploy it as a **Render Free Python Web Service** without Docker using [`backend/README.md`](backend/README.md), then paste the service URL into NutriLens under **Connect multi-food segmentation engine**. The model checkpoint is approximately 223 MB and downloads on the first segmentation request.

The website sends the selected/captured image to the public FastAPI endpoint you configure. Render Free has limited memory, CPU-only inference and idle spin-down; YOLO11l may exceed its memory allowance. This is an accuracy-first feasibility test, not a promise of production performance.

## AI models and sources

- Food matching: [Xenova/clip-vit-base-patch32](https://huggingface.co/Xenova/clip-vit-base-patch32), a CLIP ONNX model compatible with Transformers.js. The app uses zero-shot candidate matching against its food vocabulary.
- Browser inference runtime: [Transformers.js](https://huggingface.co/docs/transformers.js).
- Background removal: [IMG.LY background-removal-js](https://github.com/imgly/background-removal-js), using the IS-Net FP16 model. See the library's [configuration and model notes](https://www.npmjs.com/package/@imgly/background-removal).
- Nutrition reference: generic food-composition estimates informed by common per-100 g food reference values. See [USDA FoodData Central](https://fdc.nal.usda.gov/) for a public reference database. The manually curated values in this MVP are not a live USDA lookup and some mixed/Nepali dish entries are broad recipe averages.

## Accuracy and known limitations

- **No model can promise perfect recognition.** CLIP is a general zero-shot vision-language model, not a food-only diagnostic system. It ranks the available prompts and can still confuse similar foods, mixed dishes, ingredients, or unfamiliar presentation styles.
- A model score is a relative match score, not a calibrated probability. The user should confirm the food name before adding it to the nutrition calculation.
- The FoodSeg103 segmentation model covers 103 ingredient classes, not every Indian or Nepali dish. Regional dish suggestions are separately ranked by CLIP and are not pixel-level segmentation.
- Segmentation is not perfect: overlapping, mixed, tiny, occluded or unfamiliar foods may be missed or mislabeled. The model is trained on FoodSeg103 and is not validated as a comprehensive India/Nepal food recognizer.
- The optional API is CPU-based and may be slow or cold-start on free hosting. It is still image inference, not live video segmentation.
- Nutrition figures are estimates per 100 g. A photograph alone cannot reliably determine serving weight, hidden ingredients, cooking oil, recipe, or preparation method. Mixed Nepali dish values vary by household and restaurant.
- Do not use these estimates as medical or clinical nutrition advice.

## Licensing note

The IMG.LY background-removal JavaScript project is published under AGPL-3.0. The optional FoodSeg103 YOLO model card states Apache-2.0; review the upstream model and dataset terms before commercial use. Review its licence and your distribution obligations before using this dependency in a closed-source or commercial product. The CLIP ONNX model card lists MIT licensing. Third-party model and runtime terms apply.

## Next engineering milestones

- Evaluate a dedicated food/produce model on a test set containing corn, banana, similar fruits, vegetables, and mixed dishes.
- Add region proposals and per-item detection so multiple foods can be identified independently.
- Replace broad recipe averages with a source-linked nutrition database and regional recipe variants.
- Add portion estimation with explicit uncertainty and user correction.
- Add browser-based mask refinement tools (erase/restore brush) for difficult edges.
- Add automated browser tests for upload, model errors, PNG transparency, and serving-size recalculation.
