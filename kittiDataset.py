import csv
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image

from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive, check_integrity, calculate_md5
from torchvision.datasets.vision import VisionDataset

import pandas as pd

class KittiDataset(VisionDataset):

    data_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
    data_raw_url = data_url + "raw_data/"

    # 2011_09_26_drive_0009/2011_09_26_drive_0009_sync.zip

    scenarios = [
        "2011_09_26_drive_0001",
        "2011_09_26_drive_0002",
        "2011_09_26_drive_0005",
        "2011_09_26_drive_0009",
        "2011_09_26_drive_0011",
        "2011_09_26_drive_0013",
        "2011_09_26_drive_0014",
        "2011_09_26_drive_0017",
        "2011_09_26_drive_0018",
        "2011_09_26_drive_0048",
        "2011_09_26_drive_0051",
        "2011_09_26_drive_0056",
        "2011_09_26_drive_0057",
        "2011_09_26_drive_0059",
        "2011_09_26_drive_0060",
        "2011_09_26_drive_0084",
        "2011_09_26_drive_0091",
        "2011_09_26_drive_0093",
        "2011_09_26_drive_0095",
        "2011_09_26_drive_0096",
        "2011_09_26_drive_0104",
        "2011_09_26_drive_0106",
        "2011_09_26_drive_0113",
        "2011_09_26_drive_0117",
        "2011_09_28_drive_0001",
        "2011_09_28_drive_0002",
        "2011_09_29_drive_0026",
        "2011_09_29_drive_0071",
    ]

    resources = {
        "depth_annotations": "data_depth_annotated.zip",
        "depth_raw": "data_depth_velodyne.zip",
    }
    md5 = {
        "depth_annotations": "7d1ce32633dc2f43d9d1656a1f875e47",
        "depth_raw": "20bd6e7dc741520240a0c471392fe9df",
    }
    extracted_folders = [
        Path("dataset") / "poses",
        Path("dataset") / "sequences",
    ]
    extracted_subfolders = [ # check inside each folder inside train/* and test/*
        Path("proj_depth") / "groundtruth" / "image_02",
        Path("proj_depth") / "groundtruth" / "image_03",
        Path("proj_depth") / "velodyne_raw" / "image_02",
        Path("proj_depth") / "velodyne_raw" / "image_03",
    ]
    extracted_sequences_data = [
        Path("velodyne"),
        Path("calib.txt"),
        Path("times.txt"),
    ]

    def __init__(self, root:str, train:bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 download: bool = False,
                 remove_finished: bool = False,
                 disableExpensiveCheck: bool = False):
        super().__init__(root, transforms, transform, target_transform)
        self.train = train
        self.root = Path(root) / "kitti_dataset"
        self._location = "training" if train else "testing"
        self.remove_finished = remove_finished
        self.disableExpensiveCheck = disableExpensiveCheck
        self.shouldDownload = download
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

        if not self._raw_folder.exists():
            self._raw_folder.mkdir(parents=True)
        if not self._extracted_folder.exists():
            self._extracted_folder.mkdir(parents=True)

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        
        self.datalist, self.calib = self._parse_data()

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any, Any]: # LeftRGB, RightRGB, Velodyne, Timestamp, CalibDict
        data = self.datalist[index]
        leftRgb = Image.open(data["left_rgb"])
        leftRgb = transforms.ToTensor()(leftRgb)
        # rightRgb = Image.open(data["right_rgb"])
        # rightRgb = transforms.ToTensor()(rightRgb)
        velodyne = self.processVelodyneBinary(data["velodyne"])
        timestamp = data["timestamp"]
        calib = self.calib

        if self.transforms:
            leftRgb = self.transforms(leftRgb)
            #leftRgb, rightRgb, velodyne = self.transforms(leftRgb, rightRgb, velodyne)
        return leftRgb
        #return { "left_rgb": leftRgb, "right_rgb": rightRgb} #, "velodyne": velodyne, "timestamp": timestamp, "calib": calib }

    def __len__(self) -> int:
        return len(self.datalist)
    
    @staticmethod
    def processVelodyneBinary(velodynePath: Path) -> Any:
        return velodynePath

    def _parse_data(self) -> Tuple[List[Any], dict]:
        sequence = "00"
        sequence_path = self._extracted_raw / "dataset" / "sequences" / sequence
        dataList, calib = self._process_sequence_path(sequence_path)
        return dataList, calib
        
    def _process_sequence_path(self, sequence_path: Path) -> Tuple[List[dict], dict]:
        assert sequence_path.exists()
        assert sequence_path.is_dir()
        
        left_rgb_path = sequence_path / "image_2"
        right_rgb_path = sequence_path / "image_3"
        velodyne_path = sequence_path / "velodyne"
        timestamp_path = sequence_path / "times.txt"
        calib_path = sequence_path / "calib.txt"
        
        assert left_rgb_path.exists()
        assert right_rgb_path.exists()
        assert velodyne_path.exists()
        assert timestamp_path.exists()
        assert calib_path.exists()

        assert len(list(left_rgb_path.iterdir())) == len(list(right_rgb_path.iterdir()))
        assert len(list(left_rgb_path.iterdir())) == len(list(velodyne_path.iterdir()))

        left_rgb_list = sorted(list(left_rgb_path.iterdir()))
        right_rgb_list = sorted(list(right_rgb_path.iterdir()))
        velodyne_list = sorted(list(velodyne_path.iterdir()))

        def timeStampGenerator():
            with open(timestamp_path, "r") as f:
                for line in f:
                    yield line.strip()

        dataList = []
        for leftRgb, rightRgb, velodyne, timestamp in zip(left_rgb_list, right_rgb_list, velodyne_list, timeStampGenerator()):
            dataList.append({
                "left_rgb": leftRgb,
                "right_rgb": rightRgb,
                "velodyne": velodyne,
                "timestamp": timestamp,
            })

        # read calib file as dict of key: List[float]
        calib = {}
        with open(calib_path, "r") as f:
            for line in f:
                key, *values = line.strip().split(" ")
                key = key[:-1] # remove ":" at the end
                calib[key] = [float(value) for value in values]

        return dataList, calib


    @property
    def _raw_folder(self) -> str:
        return self.root / "downloaded"
    
    @property
    def _extracted_folder(self) -> str:
        return self.root / "extracted"
    
    @property
    def _extracted_depth(self) -> str:
        return self._extracted_folder / "depth"

    @property
    def _extracted_raw(self) -> str:
        return self._extracted_folder / "raw"

    def _check_exists(self) -> bool:

        return False
    
        for folders in self.extracted_folders:
            if not (self._extracted_raw / folders).exists():
                print("{} not found".format(self._extracted_raw / folders))
                return False
        for key, md5 in self.md5.items():
            file = self.resources[key]
            if not (self._raw_folder / file).exists():
                print("{} not found".format(file))
                return False

        for folders in self.extracted_subfolders:
            folder = self._extracted_depth / "train"
            if not folder.exists():
                print("{} not found".format(folder))
                return False
            for subfolders in folder.iterdir():
                if not (subfolders / folders).exists():
                    print("{} not found".format(subfolders / folders))
                    return False
            folder = self._extracted_depth / "val"
            if not folder.exists():
                print("{} not found".format(folder))
                return False
            for subfolders in folder.iterdir():
                if not (subfolders / folders).exists():
                    print("{} not found".format(subfolders / folders))
                    return False

        
        for folders in self.extracted_folders:
            folder = self._extracted_raw
            if not folder.exists():
                print("{} not found".format(folder))
                return False
            if not (folder / folders).exists():
                print("{} not found".format(folder / folders))
                return False
        
        folder = self._extracted_raw / "dataset" / "sequences"
        if not folder.exists():
            print("{} not found".format(folder))
            return False
        for file_folder in self.extracted_sequences_data:
            for subfolder in folder.iterdir():
                if not (subfolder / file_folder).exists():
                    print("{} not found".format(subfolder / file_folder))
                    return False

        def expensiveCheck():
            for key, md5 in self.md5.items():
                file = self.resources[key]
                if not check_integrity(self._raw_folder / file, md5):
                    print("{} doesn't have md5 {}".format(file, md5))
                    return False
            return True

        if self.shouldDownload:
            return expensiveCheck()
        if not self.disableExpensiveCheck:
            return expensiveCheck()
        return True
        

    def download(self) -> None:
        if self._check_exists():
           print("Files already downloaded and verified")
           return
        for filePrefix in self.scenarios:
            url = self.data_raw_url + file / f"{file}.zip"
            file = f"{file}.zip"
            download_and_extract_archive(url, download_root=self._raw_folder, extract_root=self._extracted_raw, filename=file, md5=None, remove_finished=self.remove_finished)
        return
        for key, md5 in self.md5.items():
            file = self.resources[key]
            if key == "depth_annotations" or key == "depth_raw":
                extracted_folder = self._extracted_depth
            else:
                extracted_folder = self._extracted_raw
            download_and_extract_archive(self.data_url + file, download_root=self._raw_folder, extract_root=extracted_folder, filename=file, md5=md5, remove_finished=self.remove_finished)