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

    return parser.parse_args()
