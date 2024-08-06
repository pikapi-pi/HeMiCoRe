import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics.pairwise import pairwise_distances, paired_cosine_distances
from utils import dump_json
# import umap.umap_ as umap


def euclidean_similarity(x, args):
    dists = pairwise_distances(x, metric='euclidean')
    dists = dists / np.max(dists)
    similarity = np.exp(-dists * args.lamb)
    return similarity

def APC(feats, args, filepath=None):
    print(feats.shape)
    umap_model = umap.UMAP(n_neighbors=15, min_dist=1, n_components=2, spread=1.0,
                           random_state=42)  # 降维到2D空间，可以根据需要更改维度
    umap_embedding = umap_model.fit_transform(feats)
    # similarity = euclidean_similarity(feats, args)
    af = KMeans(n_clusters=args.num_clusters, random_state=985).fit(umap_embedding)#AffinityPropagation(max_iter=400, convergence_iter=50, preference=args.preference, damping=args.damping, affinity="euclidean", random_state=42).fit(umap_embedding)
    labels = af.labels_

    # cluster_centers_indices = af.cluster_centers_indices_
    # n_clusters = len(cluster_centers_indices)
    # print(n_clusters)
    # while n_clusters > 10:
    #     af = AffinityPropagation(max_iter=400, convergence_iter=50, preference=None, damping=args.damping, affinity="euclidean", random_state=42).fit(cluster_centers)
    #     cluster_centers = af.cluster_centers_
    #     cluster_centers_indices = af.cluster_centers_indices_
    #     n_clusters = len(cluster_centers_indices)
    #     for i in range(len(labels)):
    #         labels[i] = af.labels_[labels[i]]


    print(np.unique(labels))
    # print(len(af_1.cluster_centers_indices))
    # y_hat = af.predict(similarity)
    # clusters = np.unique(y_hat)
    colors = ['#FFFF99', '#CCCCFF', '#FF9999', '#CC9933', '#9966FF', '#FF6666', '#FF3333', '#66FFFF',
              '#66CCFF', '#660000', '#006600', '#CCCC00', '#669966']
    # umap_model = umap.UMAP(n_neighbors=15, min_dist=1, n_components=2, spread=1.0, random_state=42)  # 降维到2D空间，可以根据需要更改维度
    # umap_embedding = umap_model.fit_transform(feats)

    # cluster_centers_embedding = af_2.cluster_centers_

    for point, label in zip(umap_embedding, labels):
        plt.scatter(point[0], point[1], color=colors[label], marker='o')
    # plt.scatter(cluster_centers_embedding[:, 0], cluster_centers_embedding[:, 1], marker='x', color='red', s=100)
    # for cluster in clusters:
    #     row_ix = np.where(y_hat == cluster)
    #     plt.scatter(umap_embedding[row_ix, 0], umap_embedding[row_ix, 1])
    plt.show()

def clustering(feats, num_clusters, filepath=None):
    k_means = KMeans(n_clusters=num_clusters, random_state=985).fit(feats)
    features_cluster_indices = np.expand_dims(k_means.labels_, axis=1)

    if filepath is not None:
        np.savez(file=filepath, features_cluster_indices=features_cluster_indices)
    return features_cluster_indices


def save_to_json(features_cluster_indices, num_clusters, filepath=None):
    cluster_features = [[] for _ in range(num_clusters)]
    for patch_idx, cluster_idx in enumerate(features_cluster_indices):
        cluster_features[cluster_idx.item()].append(patch_idx)
    if filepath is not None:
        dump_json(cluster_features, filepath)
    return cluster_features


def run(args):
    save_dir = Path(args.feat_dir) / f'k-means-{args.num_clusters}'
    save_dir.mkdir(parents=True, exist_ok=True)

    img_features_npz = sorted(list(Path(args.feat_dir).glob('*.npz')))
    img_features_npz_bar = tqdm(img_features_npz)
    for i, feat_npz in enumerate(img_features_npz):
        case_id = feat_npz.stem
        # if case_id != "1400513-HE":
        #     continue
        npz_filepath = save_dir / f'{case_id}.npz'
        json_filepath = save_dir / f'{case_id}.json'
        if npz_filepath.exists() and not args.exist_ok:
            print(f"{npz_filepath} is exists!")
            continue

        feat_dict = np.load(str(feat_npz))
        if feat_dict['img_features'].shape[0] < args.num_clusters:
            print(f"{feat_npz.stem}'s number of features < number of clusters, can't clustering.")
            continue

        # clustering and save as .npz file and .json file
        # features_cluster_indices = clustering(feat_dict['img_features'], args.num_clusters, filepath=npz_filepath)
        features_cluster_indices = clustering(feat_dict['img_features'], args.num_clusters, filepath=npz_filepath)
        save_to_json(features_cluster_indices, args.num_clusters, filepath=json_filepath)

        img_features_npz_bar.set_description(
            f"{i + 1:3}/{len(img_features_npz):3}, {feat_npz.stem}'s features: {feat_dict['img_features'].shape[0]}"
        )
        img_features_npz_bar.update()
    img_features_npz_bar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', type=str, default='/media/lzs/41e45ae4-ad45-41db-9bc2-b8299dd12de8/survival_prediction/patch/features/resnet18', help="the directory containing features files(.npz)")
    parser.add_argument('--preference', type=int, default=-10000, help='preference for AP clustering')
    parser.add_argument('--damping', type=int, default=0.9, help='damping for AP clustering')
    parser.add_argument('--num_clusters', type=int, default=10, help='the numbers of k-means clusters')
    parser.add_argument('--lamb', type=float, default=0.25)
    parser.add_argument('--exist_ok', action='store_true', default=False)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()