import os
import torch

KAGGLE_DATASET = "radiokaktus/photos-no-class-test"

BUCKET_NAME = "test-bucket-6e247f43"

BASE_DATA_DIR = "photos_no_class"

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
PROCESSING_ORDER = ["test", "val", "train"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = os.cpu_count() or 4

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')