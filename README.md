# NutriLens

**See your food differently.** A privacy-first browser food studio for image cutouts, AI-assisted food suggestions, and editable nutrition estimates.

## Current experience

- Responsive, minimal food-image workspace.
- JPG, PNG, and WEBP upload with drag-and-drop.
- Client-side background removal using IMG.LY IS-Net FP16, transparent PNG output, progress feedback, and GPU-to-CPU fallback.
- Bottom-to-top reveal animation, original/cutout preview, PNG download, and drag-out support.
- Zero-shot image-to-food matching using CLIP through Transformers.js.
- Food vocabulary spanning fruits, vegetables, grains, proteins, common meals, and Nepali dishes.
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

The first background-removal and AI-recognition runs need an internet connection to download browser libraries and model weights. Model assets are cached by the browser when supported. Images are processed in the browser; this project does not upload them to a NutriLens backend.

## AI models and sources

- Food matching: [Xenova/clip-vit-base-patch32](https://huggingface.co/Xenova/clip-vit-base-patch32), a CLIP ONNX model compatible with Transformers.js. The app uses zero-shot candidate matching against its food vocabulary.
- Browser inference runtime: [Transformers.js](https://huggingface.co/docs/transformers.js).
- Background removal: [IMG.LY background-removal-js](https://github.com/imgly/background-removal-js), using the IS-Net FP16 model. See the library's [configuration and model notes](https://www.npmjs.com/package/@imgly/background-removal).
- Nutrition reference: generic food-composition estimates informed by common per-100 g food reference values. See [USDA FoodData Central](https://fdc.nal.usda.gov/) for a public reference database. The manually curated values in this MVP are not a live USDA lookup and some mixed/Nepali dish entries are broad recipe averages.

## Accuracy and known limitations

- **No model can promise perfect recognition.** CLIP is a general zero-shot vision-language model, not a food-only diagnostic system. It ranks the available prompts and can still confuse similar foods, mixed dishes, ingredients, or unfamiliar presentation styles.
- A model score is a relative match score, not a calibrated probability. The user should confirm the food name before adding it to the nutrition calculation.
- The current analysis ranks foods for the image as a whole. It does not yet perform robust per-item instance segmentation/counting for every ingredient in a mixed plate. Multiple foods can be added after reviewing suggestions or using the food list.
- Nutrition figures are estimates per 100 g. A photograph alone cannot reliably determine serving weight, hidden ingredients, cooking oil, recipe, or preparation method. Mixed Nepali dish values vary by household and restaurant.
- Do not use these estimates as medical or clinical nutrition advice.

## Licensing note

The IMG.LY background-removal JavaScript project is published under AGPL-3.0. Review its licence and your distribution obligations before using this dependency in a closed-source or commercial product. The CLIP ONNX model card lists MIT licensing. Third-party model and runtime terms apply.

## Next engineering milestones

- Evaluate a dedicated food/produce model on a test set containing corn, banana, similar fruits, vegetables, and mixed dishes.
- Add region proposals and per-item detection so multiple foods can be identified independently.
- Replace broad recipe averages with a source-linked nutrition database and regional recipe variants.
- Add portion estimation with explicit uncertainty and user correction.
- Add browser-based mask refinement tools (erase/restore brush) for difficult edges.
- Add automated browser tests for upload, model errors, PNG transparency, and serving-size recalculation.
