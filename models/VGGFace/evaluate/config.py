import os

BASE_FOLDER_LOCAL = "../../../webface_112x112" 

NUM_WORKERS = os.cpu_count() or 4

FAISS_INDEX_FILE = "gallery_vgg.index"
FAISS_MAPPING_FILE = "gallery_id_map_vgg.json"

RESULTS_CSV = "occlusion_results_vgg.csv"
OCCLUSION_SIZE = 30
