import os

# --- Konfiguracja Ścieżek Lokalnych ---
BASE_FOLDER_LOCAL = "webface_112x112" 

# --- Ustawienia Równoległości ---
# Ustaw na liczbę rdzeni CPU (lub więcej, jeśli masz szybki dysk NVMe)
NUM_WORKERS = os.cpu_count() or 4

# --- Konfiguracja FAISS & Galerii ---
FAISS_INDEX_FILE = "gallery.index"
# ... (reszta pliku bez zmian) ...