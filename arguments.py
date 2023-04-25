import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=3, help='Number of GMM components')
    parser.add_argument('--n_ch', type=int, help='Number of channels for the mapping model')