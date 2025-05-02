import argparse
from tqdm import tqdm
import art
import os
import h5py
import cv2
from ofp.ofp import OnlineFieldPerception

def infer():
    parser = argparse.ArgumentParser(description="Train OnlineFieldPerception model")
    parser.add_argument("video_path", type=str, help="Path to 1. dir with png/jpg 2. mp4 file")
    parser.add_argument("--ckpt", type=str, default="ofp_model.pkl", help="Filename to save model checkpoint")
    parser.add_argument("--dataset", action="store_true", default=False, help="Don't rerun tracking/embeddings, just train on a already tracked dataset")
    parser.add_argument("-e", type=str, default="sam2",  help="Embeddings to use")
    parser.add_argument("--embed", action="store_true", default=False,  help="Run embeddings")

    args = parser.parse_args()

    art.tprint("Online Field Perception") # You can change font if desired

    if args.dataset:
        data_dir = args.video_path

    # Training
    embedding = None
    if args.embed:
        embedding = args.e
    model = OnlineFieldPerception(embedder=embedding)
    model.load(args.ckpt)

    # Run visualization
    vis_dir = os.path.join(data_dir, "data_products", "ofp_vis")
    os.makedirs(vis_dir, exist_ok=True)

    with h5py.File(os.path.join(data_dir, "dataset.h5"), 'r') as f:
        front = f['front'][:]
        if not args.embed:
            front_sam2 = f[f'front_{args.e}'][:]
        for i in tqdm(range(len(f["front"])), desc="Visualizing", bar_format="{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
            image = front[i]           # (720, 720, 3)
            if not args.embed:
                embedding = front_sam2[i]  # (256, 45, 45)
                trav_img = model.predict(image, embedding)
            else:
                trav_img = model.predict(image, None)
            vis = model.visualize(image, trav_img)

            cv2.imwrite(os.path.join(vis_dir, f"{i:04d}.png"), vis)
