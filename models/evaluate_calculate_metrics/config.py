import os

# --- Konfiguracja Ścieżek Lokalnych ---
# Główny folder datasetu, który zawiera podfoldery train/ val/ test/
BASE_FOLDER_LOCAL = "../../scripts/casia_dataset/webface_112x112" 
# Możesz też ustawić pełną ścieżkę, np. "C:/Users/User/Projekty/webface_112x112"

# --- Konfiguracja FAISS & Galerii ---
FAISS_INDEX_FILE = "../ArcFace_Small/gallery.index"
FAISS_MAPPING_FILE = "../ArcFace_Small/gallery_id_map.json"

# --- Konfiguracja Ewaluacji ---
RESULTS_CSV = "../../scores/ArcFace_Small/occlusion_results.csv"
OCCLUSION_SIZE = 30
VER_SCORES_CSV = "../../scores/ArcFace_Small/verification_scores.csv"

NUM_WORKERS = 4