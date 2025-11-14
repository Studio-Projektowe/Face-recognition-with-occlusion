import os

# --- Konfiguracja Ścieżek Lokalnych ---
BASE_FOLDER_LOCAL = "../../scripts/casia_dataset/webface_112x112" 

# --- Ustawienia Równoległości ---
# Ustaw na liczbę rdzeni CPU (lub więcej, jeśli masz szybki dysk NVMe)
NUM_WORKERS = os.cpu_count() or 4

# --- Konfiguracja FAISS & Galerii ---
FAISS_INDEX_FILE = "gallery_vgg.index" # Nowa nazwa, żeby nie pomylić
FAISS_MAPPING_FILE = "gallery_id_map_vgg.json" # Nowa nazwa

# --- Konfiguracja Ewaluacji ---
RESULTS_CSV = "occlusion_results_vgg.csv" # Nowa nazwa
OCCLUSION_SIZE = 30
