import os
import pickle
import numpy as np
import cv2
import h5py
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
from typing import Optional
import time


class OnlineFieldPerception:
    def __init__(self):
        self.transform = None
        self.centers = None
        self.max_dists = None

    def _extract_vectors_incremental(self, h5_path, increment=1):
        vectors = []
        non_trav_v = []
        with h5py.File(h5_path, 'r') as f:
            front_sam2_ds = f['front_sam2']
            trajs = f['Trajectories']
            img_shape = f['front'][0].shape[:2]

            for i in tqdm(range(len(f["Index"])), desc="Processing images", bar_format="{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                if i % increment != 0:
                    continue
                sensor_idx = f["Index"][i]
                embedding = front_sam2_ds[sensor_idx]  # shape (256, 45, 45)
                bool_mask = np.zeros(img_shape, dtype=np.uint8)
                bool_mask[trajs[i][:, 1], trajs[i][:, 0]] = 1
                bool_mask = cv2.resize(bool_mask, embedding.shape[1:][::-1], interpolation=cv2.INTER_NEAREST).astype(bool)

                masked = embedding[:, bool_mask]  # shape (256, P)
                non_masked = embedding[:, np.logical_not(bool_mask)]  # shape (256, P)
                vectors.append(masked.T)
                non_trav_v.append(non_masked.T)

        return np.concatenate(vectors, axis=0), np.concatenate(non_trav_v, axis=0)

    def train(self, data_dir: str, filename: str, dim: int = 100, k_clusters: int = 100, increment: int = 1):
        s = time.time()
        h5_path = os.path.join(data_dir, "dataset.h5")
        vectors, other_vectors = self._extract_vectors_incremental(h5_path, increment)

        print("Training...")
        if dim is None:
            dim = vectors.shape[1]
            self.transform = lambda x: x
        else:
            pca = PCA(n_components=dim, svd_solver='randomized')
            combined = np.concatenate([vectors, other_vectors], axis=0)
            pca.fit(combined)
            self.transform = pca.transform

        trav_reduced = self.transform(vectors)
        other_reduced = self.transform(other_vectors)
        combined = np.concatenate((trav_reduced, other_reduced))

        kmeans = KMeans(n_clusters=k_clusters, n_init='auto')
        labels = kmeans.fit_predict(combined)
        trav_clusters = np.unique(labels[:len(trav_reduced)])
        self.centers = kmeans.cluster_centers_[trav_clusters]

        self.max_dists = []
        for i, tc in enumerate(trav_clusters):
            cluster_points = combined[labels == tc]
            self.max_dists.append(np.linalg.norm(cluster_points - self.centers[i][None, :], axis=1).max())
        self.max_dists = np.array(self.max_dists)

        os.makedirs(os.path.join(data_dir, "checkpoints"), exist_ok=True)
        filepath = os.path.join(data_dir, "checkpoints", filename)
        with open(filepath, 'wb') as f:
            pickle.dump({'transform': self.transform, 'centers': self.centers, 'max_dists': self.max_dists}, f)
        print("Trained!")
        print("Total time:",f"\033[92m{time.time()-s:.2f} seconds\033[0m")
        print(filepath)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.transform = data['transform']
            self.centers = data['centers']
            self.max_dists = data['max_dists']

    def predict(self, image: np.ndarray, embedding: np.ndarray, kern: str = 'avg') -> np.ndarray:
        emb_shape = embedding.shape[1:]
        emb = embedding.reshape(256, -1).T

        emb_pca = self.transform(emb)
        dists = np.linalg.norm(emb_pca[:, None, :] - self.centers[None, :, :], axis=2)
        min_dist = np.min(dists, axis=1)

        max_dist = self.max_dists[np.argmin(dists, axis=1)]
        min_dist = np.minimum(min_dist, max_dist)

        min_dist_img = min_dist.reshape(emb_shape)
        min_dist_img_norm = cv2.normalize(min_dist_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        upscaled = cv2.resize(min_dist_img_norm, image.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        return upscaled

    def visualize(self, image: np.ndarray, trav_img: np.ndarray) -> np.ndarray:
        heatmap = cv2.applyColorMap(trav_img, cv2.COLORMAP_JET)
        return cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
