import os
from google.cloud import storage, secretmanager
from tqdm.auto import tqdm
import kagglehub
import shutil

PROJECT_ID = os.environ.get("PROJECT_ID")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
KAGGLE_DATASET = os.environ.get("KAGGLE_DATASET")
SECRET_ID = "kaggle-api-key"
SECRET_VERSION = "latest"

def access_secret_version(project_id, secret_id, version_id):
    """Pobiera sekret z Google Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

def setup_kaggle_api():
    """Konfiguruje Kaggle API używając sekretu."""
    print("Konfigurowanie Kaggle API...")
    kaggle_json_content = access_secret_version(PROJECT_ID, SECRET_ID, SECRET_VERSION)
    
    kaggle_dir = os.path.expanduser("~/.kaggle") 
    os.makedirs(kaggle_dir, exist_ok=True)
    
    key_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(key_path, "w") as f:
        f.write(kaggle_json_content)
        
    os.chmod(key_path, 0o600)
    print("Kaggle API skonfigurowane.")

def download_and_upload_files(bucket_name, dataset_name):
    """
    Pobiera i rozpakowuje dataset za pomocą kagglehub,
    a następnie wysyła rozpakowane pliki do GCS.
    """
    downloaded_path = ""
    try:
        print(f"Rozpoczynam pobieranie i rozpakowywanie zbioru: {dataset_name} (to potrwa...)")
        downloaded_path = kagglehub.dataset_download(dataset_name)
        print(f"Zbiór pobrany i rozpakowany do: {downloaded_path}")

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        print(f"Rozpoczynam wysyłanie plików do wiadra: {bucket_name}")
        
        file_count = 0
        
        for root, _, files in os.walk(downloaded_path):
            for filename in tqdm(files, desc="Przesyłanie plików", leave=False):
                    
                local_path = os.path.join(root, filename)
                
                blob_name = os.path.relpath(local_path, start=downloaded_path)
                
                blob = bucket.blob(blob_name)
                
                blob.upload_from_filename(local_path)
                file_count += 1

        print(f"\nPrzesłano łącznie {file_count} plików.")
        if file_count == 0:
            print("UWAGA: Nie przesłano żadnych plików. Sprawdź, czy kagglehub poprawnie pobrał zbiór.")

    except Exception as e:
        print(f"WYSTĄPIŁ BŁĄD: {e}")
        raise e
    finally:
        if downloaded_path and os.path.exists(downloaded_path):
            print(f"Sprzątanie dysku tymczasowego... (usuwanie {downloaded_path})")
            base_cache_dir = os.path.expanduser("~/.cache/kagglehub")
            if os.path.exists(base_cache_dir):
                 shutil.rmtree(base_cache_dir)
                 print("Usunięto folder cache.")


if __name__ == "__main__":
    if not PROJECT_ID or not BUCKET_NAME:
        print("BŁĄD: Zmienne środowiskowe PROJECT_ID i BUCKET_NAME są wymagane!")
        exit(1)
        
    setup_kaggle_api()
    download_and_upload_files(BUCKET_NAME, KAGGLE_DATASET)
    print("--- ZROBIONE! ---")