# NutriLens

**See your food differently.** NutriLens is a browser-based food image workspace that combines transparent background removal with food recognition and an editable nutrition estimate.

## Current MVP

- Upload, choose, or drag-and-drop JPG, PNG, and WEBP images (up to 10 MB).
- Remove image backgrounds and export a transparent PNG.
- Smooth bottom-to-top reveal animation.
- Drag the isolated image from the preview into compatible apps.
- Detect a limited set of common food categories with an in-browser object-detection model.
- Add foods manually when the model cannot recognize them.
- Adjust serving sizes and calculate estimated calories, protein, carbohydrates, and fat.
- Keep a small analysis history in browser LocalStorage.
- Responsive desktop and mobile layout.

## Run locally

No build step is required for the static MVP.

1. Clone this repository.
2. Open `index.html` through a local static server (recommended) or deploy the repository with GitHub Pages.
3. For example, with Python installed:

   ```bash
   python -m http.server 8000
   ```

4. Visit `http://localhost:8000`.

The first use of background removal and food recognition requires an internet connection to load the open-source browser libraries and model assets. The image processing runs in the browser; NutriLens does not provide an image-upload backend.

## Free/open-source browser dependencies

- [@imgly/background-removal](https://github.com/imgly/background-removal-js) for client-side image background removal.
- [Transformers.js](https://github.com/huggingface/transformers.js) with the DETR ResNet-50 model for object detection.
- [Google Fonts](https://fonts.google.com/) for typography (the app falls back to system fonts).

Model weights and runtime assets are downloaded from their respective CDNs on first use. Hosting and usage policies of those services apply.

## Important limitations

- **No 100% calorie accuracy:** a 2D image cannot establish exact food weight, hidden ingredients, cooking oil, recipe, or preparation method.
- The initial object-detection model recognizes only a limited set of common COCO classes that overlap with food (for example apple, banana, orange, broccoli, carrot, sandwich, pizza, hot dog, donut, and cake). It is not a specialist multi-food segmentation model.
- Nutrition values are generic reference estimates per 100 g. Serving weights begin at 100 g and must be checked or edited by the user.
- Multiple overlapping foods, regional dishes, mixed meals, and small ingredients may be missed or misidentified.
- This is an informational prototype, not medical advice or a clinical nutrition tool.

## Roadmap

- Add a dedicated food segmentation model that can identify multiple overlapping foods.
- Improve edge refinement and transparent cutout quality testing.
- Expand nutrition references with cited sources and regional foods.
- Add confidence-aware portion estimation and explicit user confirmation.
- Add exportable nutrition reports and a meal diary.

## License

MIT
