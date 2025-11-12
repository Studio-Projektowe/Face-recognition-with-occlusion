import os

# --- Konfiguracja GCS ---
# Upewnij się, że masz ustawioną zmienną środowiskową GOOGLE_APPLICATION_CREDENTIALS
# wskazującą na plik JSON z kluczem serwisowym.
# Lub zaloguj się przez `gcloud auth application-default login`

# Nazwa Twojego bucketu
BUCKET_NAME = "test-bucket-6e247f43" 

# Ścieżka bazowa do folderu testowego w GCS
BASE_FOLDER_GCS = "photos_no_class/test"

# Gdzie skrypty mają tymczasowo pobierać pliki
LOCAL_DATA_DIR = "./gcs_temp_cache"

# --- Konfiguracja FAISS & Galerii ---
# Pliki wyjściowe dla Twojego "modelu" (indeksu galerii)
FAISS_INDEX_FILE = "gallery.index"
FAISS_MAPPING_FILE = "gallery_id_map.json" # Mapuje indeks FAISS (np. 0) na ID ('id_1234')

# --- Konfiguracja Ewaluacji ---
# Plik CSV z wynikami
RESULTS_CSV = "occlusion_results.csv"

# Rozmiar "paska" okluzji w pikselach
OCCLUSION_SIZE = 30