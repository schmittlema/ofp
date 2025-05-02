import argparse
import art
import os
import h5py
import cv2
from ofp.ofp import OnlineFieldPerception

def train():
    parser = argparse.ArgumentParser(description="Train OnlineFieldPerception model")
    parser.add_argument("data_dir", type=str, help="Path to directory containing dataset.h5")
    parser.add_argument("--filename", type=str, default="traversability_model.pkl", help="Filename to save model checkpoint")
    parser.add_argument("--dim", type=int, default=100, help="Number of PCA dimensions")
    parser.add_argument("--k_clusters", type=int, default=100, help="Number of KMeans clusters")
    parser.add_argument("--increment", type=int, default=1, help="Increment for image sampling")
    parser.add_argument("--kern", type=str, default="", help="Kernel for postprocessing ('', 'avg', 'gaus')")
    parser.add_argument("--viz", action="store_true", help="Run visualization on training data after training")
    args = parser.parse_args()

    art.tprint("Online Field Perception") # You can change font if desired

    model = OnlineFieldPerception()
    model.train(
        data_dir=args.data_dir,
        filename=args.filename,
        dim=args.dim,
        k_clusters=args.k_clusters,
        increment=args.increment,
    )

    # Run visualization
    if args.viz:
        print("Running visualization...")
        vis_dir = os.path.join(args.data_dir, "data_products", "ofp_vis")
        os.makedirs(vis_dir, exist_ok=True)

        with h5py.File(os.path.join(args.data_dir, "dataset.h5"), 'r') as f:
            for i in range(len(f["front"])):
                image = f["front"][i]           # (720, 720, 3)
                embedding = f["front_sam2"][i]  # (256, 45, 45)

                trav_img = model.predict(image, embedding, kern=args.kern)
                vis = model.visualize(image, trav_img)

                cv2.imwrite(os.path.join(vis_dir, f"{i:04d}.png"), vis)
