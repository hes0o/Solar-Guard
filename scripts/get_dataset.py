"""Download the Kaggle 'Solar Panel Images' dataset into data/raw (no API token needed)."""
import kagglehub

print(kagglehub.dataset_download("pythonafroz/solar-panel-images", output_dir="data/raw"))
