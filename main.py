from arguments import commandParser
from slamNet import SlamNet
import torch
from kittiDataset import KittiDataset

from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

# find the huber loss between the estimated pose and the ground truth pose
# The value of delta can be altered
def huber_loss(pose_estimated, actual_pose, delta = 0.1):
    residual = torch.abs(pose_estimated - actual_pose)
    is_small_res = residual < delta
    return torch.where(is_small_res, 0.5 * residual ** 2, delta * (residual - 0.5 * delta))


def lossfunction():
    return huber_loss

def optimizerfunction(net, lr):
    return optim.SGD(net.parameters(), lr=lr)

def validation(model, dataLoader, criterion):
    totalLoss = 0.0
    with tqdm(total = len(dataLoader), desc=f"Validation", position=2) as batchBar:
        for imagePrev, image, pose in dataLoader:
            output = model(imagePrev, image)
            loss = criterion(output, pose)
            totalLoss += loss.item()

        batchBar.desc = f"Validation loss: {totalLoss / len(dataLoader)}"

    return totalLoss

def train(arg, slamNet):
    train_data = KittiDataset(arg.dataset_path, download=arg.download_dataset, disableExpensiveCheck=True)
    dataLoader = DataLoader(train_data, batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers, pin_memory=True)

    validation_data = KittiDataset(arg.dataset_path, download=arg.download_dataset, train=False, disableExpensiveCheck=True,
                                   validation=True)
    validation_dataLoader = DataLoader(validation_data, batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers, pin_memory=True)

    print(f"Number of training data: {len(train_data)}")
    print(f"Number of validation data: {len(validation_data)}")

    criterion = lossfunction()
    optimizer = optimizerfunction(slamNet, arg.lr)

    decay_step = 0

    with tqdm(total = arg.epochs, desc=f"Epochs", position=0) as epochBar:
        for epoch in arg.epochs:
            epochLoss = 0.0
            runningLoss = 0.0
            with tqdm(total = len(dataLoader), desc=f"Epoch {epoch} / {arg.epochs}", position=1) as batchBar:
                for i, (imagePrev, image, pose) in enumerate(dataLoader, 0):

                    optimizer.zero_grad()

                    output = slamNet(imagePrev, image)
                    loss = criterion(output, pose)
                    loss.backward()

                    optimizer.step()

                    batchBar.update(imagePrev.shape[0])

                    runningLoss += loss.item()
                    epochLoss += loss.item()
                    if i % 2000 == 1999:
                        batchBar.desc = f"Epoch {epoch} / {arg.epochs}, loss: {runningLoss / 2000}"
                        runningLoss = 0.0
            epochBar.desc = f"Epoch {epoch} / {arg.epochs}, average training loss: {epochLoss / len(dataLoader)}"

            validationLoss = validation(slamNet, validation_dataLoader, criterion)

            if epoch + 1 % arg.decay_step == 0:
                decay_step += 1
                optimizer = optimizerfunction(slamNet, lr=arg.lr * (arg.decay_rate ** decay_step))

            epochBar.update(1)
    torch.save(slamNet.state_dict(), arg.save_model)

def test(arg, model, model_file):
    model.load_state_dict(torch.load(model_file))
    testData = KittiDataset(arg.dataset_path, download=arg.download_dataset, train=False, disableExpensiveCheck=True)
    dataLoader = DataLoader(testData, batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers, pin_memory=True)


def main(arg):
    expected_shape = (arg.batch_size, 4, 80, 80)
    slamNet = SlamNet(expected_shape, arg.num_particles, is_training=arg.is_training, is_pretrain_obs=arg.is_pretrain_obs, is_pretrain_trans= arg.is_pretrain_trans).cuda()
    if arg.test_only:
        test(arg, slamNet, arg.load_model)
        return

    train(arg, slamNet)
    test(arg, slamNet, arg.save_model)


if __name__ == "__main__":
    arg = commandParser()
    main(arg)

