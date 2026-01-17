# ArcFace Small (buffalo_s)

## Model tested
**ArcFace Small (buffalo_s)** from InsightFace — a lightweight ArcFace model (MobileFaceNet-equivalent) optimized for speed.

## Evaluation setup
- Gallery/probe split from the test set; gallery embeddings averaged per identity.
- Probe images were synthetically occluded with a **20px black bar** centered on the eye region computed from landmarks, and spanning the face width defined by `bbox`.
- Embeddings are extracted from the **full image** using InsightFace face detection (no explicit bbox crop in this script).
- Similarity search used cosine similarity with FAISS (top-$k=3$).

## Result
- **Rank-1:** 86.65%
- **Rank-3:** 90.68%

## Testing mistakes / limitations that could affect the score
- **Missing metadata leads to skipping samples:** if a JSON file lacked `landmarks` or `bbox`, the sample was skipped entirely, which may bias results toward cleaner images.
- **Fixed 20px bar size:** the occlusion height did not scale with face size or resolution, so smaller/larger faces were occluded disproportionately.
- **Occlusion color mismatch:** training augmentations used random-colored bars, but evaluation used a deterministic black bar; this distribution shift could slightly bias performance in either direction.
