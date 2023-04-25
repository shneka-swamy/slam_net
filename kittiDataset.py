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

    filter_scenarios = [
        "2011_09_29",
    ]

    resources = {
            "data_depth_annotated.zip": "7d1ce32633dc2f43d9d1656a1f875e47",
            "data_depth_velodyne.zip": "20bd6e7dc741520240a0c471392fe9df",
    }

    def __init__(self, root:str, train:bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 download: bool = False,
                 remove_finished: bool = False,
                 disableExpensiveCheck: bool = False):
        super().__init__(root, transforms, transform, target_transform)
        self.train = train
        self.root = Path(Path(root) / "kitti_dataset")
        self._location = "training" if train else "testing"
        self.remove_finished = remove_finished
        self.disableExpensiveCheck = disableExpensiveCheck
        self.shouldDownload = download
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

        if not self._download_folder.exists():
            self._download_folder.mkdir(parents=True)
        if not self._extracted_folder.exists():
            self._extracted_folder.mkdir(parents=True)

        self.scenariosFile = Path(self.root) / "kittiMd5.txt"
        self.scenarios = self._getScenarios(self.scenariosFile, self.filter_scenarios)

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        datalist, calib = self._parse_data()
        self.datalist = []
        for data in datalist:
            self.datalist.append({"leftRgb":data["leftRgb"], "calib":calib})

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = self.datalist[index]
        leftRgb = Image.open(data["leftRgb"])
        leftRgb = transforms.ToTensor()(leftRgb)
        calib = data["calib"]

        if self.transforms:
            leftRgb = self.transforms(leftRgb)
        return leftRgb, calib

    def __len__(self) -> int:
        return len(self.datalist)

    def _parse_data(self) -> Tuple[List[Any], dict]:
        listImages = []
        calib = {}
        for folder_file, _ in self.scenarios:
            if folder_file.endswith("calib.zip"):
                folder_prefix = '_'.join(folder_file.split('_')[:-1])
                calibFile = self._extracted_raw / folder_prefix / "calib_cam_to_cam.txt"
                calib['cam2cam'] = self._calib_to_dict(calibFile)
                calibFile = self._extracted_raw / folder_prefix / "calib_velo_to_cam.txt"
                calib['velo2cam'] = self._calib_to_dict(calibFile)
                calibFile = self._extracted_raw / folder_prefix / "calib_imu_to_velo.txt"
                calib['imu2velo'] = self._calib_to_dict(calibFile)
                continue
            folder_prefix = '_'.join(folder_file.split('_')[:-3])
            folder = folder_file.split('.')[0]
            folder = self._extracted_raw / folder_prefix / folder / "image_02" / "data"
            for file in folder.iterdir():
                listImages.append({"leftRgb":file})
        return listImages, calib

    @property
    def _download_folder(self) -> Path:
        return self.root / "downloaded"

    @property
    def _extracted_folder(self) -> Path:
        return self.root / "extracted"

    @property
    def _extracted_depth(self) -> Path:
        return self._extracted_folder / "depth"

    @property
    def _extracted_raw(self) -> Path:
        return self._extracted_folder / "raw"

    def _calib_to_dict(self, calibFile: Path) -> dict:
        calib = {}
        with open(calibFile, "r") as f:
            for line in f:
                key, *values = line.strip().split(" ")
                key = key[:-1] # remove ":" at the end
                if key == 'calib_time':
                    continue
                calib[key] = [float(value) for value in values]
        return calib


    def _getScenarios(self, scenariosFile, scenariosFilter) -> List[Tuple[str, str]]:
        if not scenariosFile.exists():
            assert False
        with open(scenariosFile, "r") as f:
            file, md5 = zip(*[line.strip().split(" ") for line in f])
            if scenariosFilter is None or len(scenariosFilter) == 0:
                return list(zip(file, md5))
            filterFiles = [f for f in file if any([f.startswith(s) for s in scenariosFilter])]
            file, md5 = zip(*[(f, m) for f, m in zip(file, md5) if f in filterFiles])
        return list(zip(file, md5))

    def _check_exists(self) -> bool:
        if not self._extracted_folder.exists():
            print(f"{self._extracted_folder} doesn't exist")
            return False
        if not self._extracted_depth.exists():
            print(f"{self._extracted_depth} doesn't exist")
            return False
        if not self._extracted_raw.exists():
            print(f"{self._extracted_raw} doesn't exist")
            return False

        if not (self._extracted_depth).exists():
            print(f"{self._extracted_depth} doesn't exist")
            return False
        for file, _ in self.scenarios:
            if file[-9:] != "calib.zip":
                folder_prefix = '_'.join(file.split('_')[:-3])
                folder = file.split('.')[0]
                if not (self._extracted_raw / folder_prefix / folder).exists():
                    print(f"{self._extracted_raw / folder_prefix / folder} doesn't exist")
                    return False
            else:
                folder_prefix = '_'.join(file.split('_')[:-1])
                expected_files = ['calib_cam_to_cam.txt', 'calib_imu_to_velo.txt', 'calib_velo_to_cam.txt']
                for expected_file in expected_files:
                    if not (self._extracted_raw / folder_prefix / expected_file).exists():
                        print(f"{self._extracted_raw / folder_prefix / expected_file} doesn't exist")
                        return False

        def expensiveCheck(dictFileMd5, folder):
            download_folder = self._download_folder;
            for file, md5 in dictFileMd5:
                if not check_integrity(str(download_folder / file), md5):
                    print("{} doesn't have md5 {}".format(file, md5))
                    return False
            return True

        if self.shouldDownload:
            return expensiveCheck(self.resources.items(), self._extracted_depth) and expensiveCheck(self.scenarios, self._extracted_raw)
        if not self.disableExpensiveCheck:
            return expensiveCheck(self.resources.items(), self._extracted_depth) and expensiveCheck(self.scenarios, self._extracted_raw)
        return True

    def _generate_url(self, file) -> str:
        raw_suffix = ["_calib.zip", "_sync.zip", "_tracklets.zip", "_extract.zip"]
        if any(suffix in file for suffix in raw_suffix):
            if file.endswith("_calib.zip"):
                return f"{self.data_raw_url}{file}"
            prefixFile = "_".join(file.split("_")[:-1])
            return f"{self.data_raw_url}{prefixFile}/{file}"
        return f"{self.data_url}{file}"

    def download(self) -> None:
        if self._check_exists():
           print("Files already downloaded and verified")
           return
        for file, md5 in self.scenarios:
            url = self._generate_url(file)
            download_folder = str(self._download_folder)
            extract_folder = str(self._extracted_raw)
            print(f"Downloading {url} to {download_folder} and extracting to {extract_folder}")
            download_and_extract_archive(url, download_root=download_folder, extract_root=extract_folder, filename=file, md5=md5, remove_finished=self.remove_finished)
        for file, md5 in self.resources.items():
            url = self._generate_url(file)
            download_folder = str(self._download_folder)
            extract_folder = str(self._extracted_depth)
            download_and_extract_archive(url, download_root=download_folder, extract_root=extract_folder, filename=file, md5=md5, remove_finished=self.remove_finished)

