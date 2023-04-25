from arguments import commandParser
from slamNet import SlamNet
import torch
from kittiDataset import KittiDataset

from torchvision import transforms


def main(arg):

    dataset = KittiDataset(arg.dataset_path, download=False, disableExpensiveCheck=True)
    dataLoader = torch.utils.data.DataLoader(dataset,
                                             batch_size=arg.batch_size,
                                             shuffle=False,
                                             num_workers=1, #arg.num_workers,
                                             pin_memory=True)

    leftRGB1, calib = next(iter(dataLoader))
    leftRGB2, calib = next(iter(dataLoader))

    for key in calib.keys():
        for k in calib[key].keys():
            print(f"calib [{key}][{k}]: {len(calib[key][k])}")

    slamNet = SlamNet(leftRGB1.shape[1:], arg.num_particles).cuda()
    output = slamNet(leftRGB1, leftRGB2)


if __name__ == "__main__":
    arg = commandParser()
    main(arg)

