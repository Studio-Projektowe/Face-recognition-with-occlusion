import os

# --- Konfiguracja Ścieżek Lokalnych ---
# Główny folder datasetu, który zawiera podfoldery train/ val/ test/
BASE_FOLDER_LOCAL = "webface_112x112" 
# Możesz też ustawić pełną ścieżkę, np. "C:/Users/User/Projekty/webface_112x112"

# --- Konfiguracja FAISS & Galerii ---
FAISS_INDEX_FILE = "gallery.index"
FAISS_MAPPING_FILE = "gallery_id_map.json"

# --- Konfiguracja Ewaluacji ---
RESULTS_CSV = "occlusion_results.csv"
OCCLUSION_SIZE = 30