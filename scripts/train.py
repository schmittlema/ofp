import argparse
from ofp import OnlineTraversability

def train():
    parser = argparse.ArgumentParser(description="Train OnlineTraversability model")
    parser.add_argument("data_dir", type=str, help="Path to directory containing dataset.h5")
    parser.add_argument("--filename", type=str, default="traversability_model.pkl", help="Filename to save model checkpoint")
    parser.add_argument("--dim", type=int, default=100, help="Number of PCA dimensions")
    parser.add_argument("--k_clusters", type=int, default=100, help="Number of KMeans clusters")
    parser.add_argument("--increment", type=int, default=1, help="Increment for image sampling")
    args = parser.parse_args()

    model = OnlineTraversability()
    model.train(data_dir=args.data_dir, filename=args.filename, dim=args.dim, k_clusters=args.k_clusters, increment=args.increment)
