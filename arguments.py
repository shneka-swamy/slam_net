import argparse
from pathlib import Path

def commandParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=3, help='Number of GMM components')
    parser.add_argument('--n-ch', type=int, help='Number of channels for the mapping model')
    parser.add_argument('--dataset-path', type=Path, help='Path to the dataset', default=Path('datasets/'))
    parser.add_argument('--num-particles', default = 128, type=int, help='Number of particles')
    parser.add_argument('--batch-size', default = 16, type=int, help='Batch size')
    parser.add_argument('--num-workers', default = 4, type=int, help='Number of workers')
    parser.add_argument('--download-dataset', action='store_true', help='should download dataset')
    parser.add_argument('--is-training', action='store_true', help='is training')
    parser.add_argument('--is-pretrain-obs', action='store_true', help='is pretraining observation model')
    parser.add_argument('--is-pretrain-trans', action='store_true', help='is pretraining transition model')

    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--epochs', default=12, type=int, help='Number of epochs')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate')
    parser.add_argument('--decay-rate', default=0.5, type=float, help='Initial learning rate')
    parser.add_argument('--decay-step', default=4, type=int, help='Decay step')

    parser.add_argument('--test-only', action='store_true', help='Test only')
    maxSaveNumber = -1
    for files in Path('model').glob('slamNet_v*.pth'):
        maxSaveNumber = max(maxSaveNumber, int(files.stem[8:]))
    idxSavedFiles = maxSaveNumber + 1
    parser.add_argument('--save-model', default=f'model/slamNet_v{idxSavedFiles:04d}.pth', type=str, help='Path to save the model')
    parser.add_argument('--load-model', default=f'model/slamNet_v{maxSaveNumber:04d}.pth', type=Path, help='Path to load the model')

    return parser.parse_args()
