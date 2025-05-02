import argparse
from tqdm import tqdm
import art
import os
import h5py
import cv2
from ofp.ofp import OnlineFieldPerception
from scripts.track import track
import scripts.data_products as dp
import time

def train():
    parser = argparse.ArgumentParser(description="Train OnlineFieldPerception model")
    parser.add_argument("video_path", type=str, help="Path to 1. dir with png/jpg 2. mp4 file")
    parser.add_argument("--filename", type=str, default="ofp_model.pkl", help="Filename to save model checkpoint")
    parser.add_argument("--dim", type=int, default=100, help="Number of PCA dimensions")
    parser.add_argument("--k_clusters", type=int, default=100, help="Number of KMeans clusters")
    parser.add_argument("-i", type=int, default=1, help="Increment for image sampling")
    parser.add_argument("--kern", type=str, default="", help="Kernel for postprocessing ('', 'avg', 'gaus')")
    parser.add_argument("--viz", action="store_true", default=False, help="Run visualization on training data after training")
    parser.add_argument("--dataset", action="store_true", default=False, help="Don't rerun tracking/embeddings, just train on a already tracked dataset")
    parser.add_argument(
        "--topic",
        type=str,
        default="/insta360/front/compressed",
        help="Topic to extract, only if bag provided",
    )
    start = time.time()
    args = parser.parse_args()

    art.tprint("Online Field Perception") # You can change font if desired

    # Tracking
    if not args.dataset:
        track_args = argparse.Namespace(
            video_path=args.video_path,
            reuse_tracks=False,
            mp4=False, # Only for nested
            frames=False, # Only for nested
            endpoint=False,
            tartan=False,
            fps=24,
            no_viz=False, # its the opposite
            topic=args.topic,
            s=0.25,
            compress_images=False,
        )
        data_dir = track(track_args)

        # Embeddings
        dp.add_embeddings(data_dir, "sam2", 0, True, True, False)
    else:
        data_dir = args.video_path

    # Training
    model = OnlineFieldPerception()
    model.train(
        data_dir=data_dir,
        filename=args.filename,
        dim=args.dim,
        k_clusters=args.k_clusters,
        increment=args.i,
    )

    # Run visualization
    if args.viz:
        vis_dir = os.path.join(data_dir, "data_products", "ofp_vis")
        os.makedirs(vis_dir, exist_ok=True)

        with h5py.File(os.path.join(data_dir, "dataset.h5"), 'r') as f:
            front = f['front'][:]
            front_sam2 = f['front_sam2'][:]
            for i in tqdm(range(len(f["front"])), desc="Visualizing", bar_format="{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                image = front[i]           # (720, 720, 3)
                embedding = front_sam2[i]  # (256, 45, 45)

                trav_img = model.predict(image, embedding, kern=args.kern)
                vis = model.visualize(image, trav_img)

                cv2.imwrite(os.path.join(vis_dir, f"{i:04d}.png"), vis)

    print("Total time:",f"\033[92m{time.time()-start:.2f} seconds\033[0m")
