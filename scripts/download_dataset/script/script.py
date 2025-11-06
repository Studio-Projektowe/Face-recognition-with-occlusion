import os
from kaggle.api.kaggle_api_extended import KaggleApi

username = os.getenv("KAGGLE_USERNAME")
key = os.getenv("KAGGLE_KEY")

os.environ["KAGGLE_USERNAME"] = username
os.environ["KAGGLE_KEY"] = key

api = KaggleApi()
api.authenticate()

print("Downloading CASIA-WebFace...")
api.dataset_download_files("debarghamitraroy/casia-webface", path="/workspace", unzip=True)

print("Uploading to GCS...")
os.system("gsutil -m cp -r /workspace/* gs://$TARGET_BUCKET/")
print("Done!")
