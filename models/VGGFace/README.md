# VGGFace (ResNet50)

## Model tested
**VGGFace (ResNet50)** — a legacy face recognition model used for historical comparison.

## Evaluation setup
- Gallery/probe split from the test set; gallery embeddings averaged per identity.
- **Gallery embeddings**: face crops are taken from `bbox` with 20% padding and resized to 224x224. If `bbox` is missing, the full image is resized to 224x224.
- **Probe embeddings**: a **20px black bar** is applied on the full image at the eye level computed from landmarks, spanning the face width from `bbox`. The occluded image is then cropped by `bbox` (with padding) and resized to 224x224.
- Similarity search used cosine similarity with FAISS (top-$k=3$).

## Result
- **Rank-1:** 39.64%
- **Rank-3:** 57.01%

## Testing mistakes / limitations that could affect the score
- **Missing metadata leads to skipping samples:** in probe evaluation, if a JSON file lacked `landmarks` or `bbox`, the sample was skipped entirely, which may bias results toward cleaner images.
- **Fixed 20px bar size:** the occlusion height did not scale with face size or resolution, so smaller/larger faces were occluded disproportionately.
- **Occlusion color mismatch:** training augmentations used random-colored bars, but evaluation used a deterministic black bar; this distribution shift could slightly bias performance in either direction.
