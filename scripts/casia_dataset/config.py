import os
import torch

KAGGLE_DATASET = "yakhyokhuja/webface-112x112"

BUCKET_NAME = "casia_prepared_dataset-6e247f43"

BASE_DATA_DIR = "webface-112x112"

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
PROCESSING_ORDER = ["test", "val", "train"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = os.cpu_count() or 4

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')