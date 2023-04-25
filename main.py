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

    imagePrev1, image1, pose1 = next(iter(dataLoader))
    imagePrev2, image2, pose2 = next(iter(dataLoader))

    print(f"imagePrev1: {imagePrev1.shape}, image1: {image1.shape}, pose1.x: {pose1['x'].shape}, pose1.y: {pose1['y'].shape}, pose1.yaw: {pose1['yaw'].shape}")
    print(f"imagePrev2: {imagePrev2.shape}, image2: {image2.shape}, pose2.x: {pose2['x'].shape}, pose2.y: {pose2['y'].shape}, pose2.yaw: {pose2['yaw'].shape}")

    slamNet = SlamNet(imagePrev1.shape[1:], arg.num_particles).cuda()
    output = slamNet(imagePrev1, image1)


if __name__ == "__main__":
    arg = commandParser()
    main(arg)

