import os
from google.cloud import storage
import io

def download_data_from_gcs(gcs_uri: str, local_path: str):
    # parse gs://bucket-name/path/to/file.csv
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name, blob_path = parts[0], parts[1]

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)

try:
    #os.mkdir(os.getcwd() + "/data")
    os.mkdir('/tmp/data/')
except:
    pass

download_data_from_gcs("gs://japanese_english_nmt/dev.en", "/tmp/data/dev.en")
download_data_from_gcs("gs://japanese_english_nmt/dev.ja", "/tmp/data/dev.ja")
download_data_from_gcs("gs://japanese_english_nmt/train.en", "/tmp/data/train.en")
download_data_from_gcs("gs://japanese_english_nmt/train.ja", "/tmp/data/train.ja")
download_data_from_gcs("gs://japanese_english_nmt/test.en", "/tmp/data/test.en")
download_data_from_gcs("gs://japanese_english_nmt/test.ja", "/tmp/data/test.ja")



# print(f"Data downloaded to: {os.getcwd()}/data/")
print("Data downloaded to: /tmp/data/")
print(f"Contents of /tmp/data/: {os.listdir('/tmp/data/')}")
