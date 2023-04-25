from arguments import commandParser
from slamNet import SlamNet
import torch
from kittiDataset import KittiDataset

from torchvision import transforms


def main(arg):

    dataset = KittiDataset(arg.dataset_path, download=True, disableExpensiveCheck=True)
    dataLoader = torch.utils.data.DataLoader(dataset,
                                             batch_size=arg.batch_size,
                                             shuffle=False,
                                             num_workers=1, #arg.num_workers,
                                             pin_memory=True)
    
    leftRGB1 = next(iter(dataLoader))
    leftRGB2 = next(iter(dataLoader))

    slamNet = SlamNet(leftRGB1.shape[1:], arg.num_particles).cuda()
    output = slamNet(leftRGB1, leftRGB2)
    

if __name__ == "__main__":
    arg = commandParser()
    main(arg)   