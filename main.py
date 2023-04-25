from arguments import commandParser
from slamNet import SlamNet
import torch
from kittiDataset import KittiDataset

from torchvision import transforms


def main(arg):

    dataset = KittiDataset(arg.dataset_path, download=False, disableExpensiveCheck=True)
    dataLoader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1, #arg.batch_size,
                                             shuffle=False,
                                             num_workers=1, #arg.num_workers,
                                             pin_memory=True)
    
    leftRGB, rightRGB, velodyne, timestamp, calibDict = next(iter(dataLoader))

    print("leftRGB.shape: {}".format(leftRGB.shape))
    print("rightRGB.shape: {}".format(rightRGB.shape))
    print("velodyne: {}".format(velodyne))
    print("timestamp: {}".format(timestamp))
    print("calibDict: {}".format(calibDict))

    leftRGB.show()
    rightRGB.show()

    batch_size = 16
    dummy_input_1 = torch.zeros((batch_size, 3, 90, 160), dtype=torch.uint8, device='cuda')
    dummy_input_2 = torch.zeros((batch_size, 3, 90, 160), dtype=torch.uint8, device='cuda')
    slamNet = SlamNet(dummy_input_1.shape[1:], arg.num_particles).cuda()
    output = slamNet(dummy_input_1, dummy_input_2)
    

if __name__ == "__main__":
    arg = commandParser()
    main(arg)   