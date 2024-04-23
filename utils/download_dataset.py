import os

import s3fs
import zipfile


def download_dataset_if_needed(path: str = "./LivingRoom_preprocessed_hack"):
    if os.path.exists(path):
        print(f"path: {path} already exist, ignorign dataset downloading")
        return

    s3_endpoint_url = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": s3_endpoint_url})

    path_to_dataset = fs.ls("gvimont/diffusion/hackathon-minarm-2024/Acoustique")[0]
    print(path_to_dataset)

    fs.download(path_to_dataset, ".")

    with zipfile.ZipFile(f"./{os.path.basename(path_to_dataset)}", "r") as zip_file:
        zip_file.extractall(path)
