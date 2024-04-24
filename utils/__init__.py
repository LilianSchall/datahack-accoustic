import os

import s3fs
import zipfile

import numpy as np


from datasets.dataset import Dataset
from datasets import livingroom
from datasets.roomsetup import RoomSetup


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
        zip_file.extractall('.')

def get_dataset(centroid_filename, deconvolved_filename, dataset_path):
    centroid, RIRs = None, None
    if centroid_filename is not None:
        centroid = np.load(os.path.join(dataset_path, centroid_filename))
        print("Shape of Centroid:")
        print(centroid.shape)

    if deconvolved_filename is not None:
        #Loading Room Impulse Response (1000 human locations x 10 microphones x M time samples)
        RIRs = np.load(os.path.join(dataset_path, deconvolved_filename), mmap_mode='r')
        print("Shape of RIRs:")
        print(RIRs.shape)

    dr = Dataset(RoomSetup(livingroom.speaker_xyz,
                    livingroom.mic_xyzs,
                    livingroom.x_min,
                    livingroom.x_max,
                    livingroom.y_min,
                    livingroom.y_max,
                    livingroom.walls), dataset_path)
    return dr, centroid, RIRs