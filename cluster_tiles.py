from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler

import statistics

NUM_CLUSTERS = 5
NUM_ELEMENTS = 10

num_pos, num_neg = 20, 20

def clustered_sampling(embeddings_file, label_tag, coords_file):
    embeddings = []
    with h5py.File(embeddings_file, "r") as f:
        for slide in f.values():
            if num_pos <= 0 and num_neg <= 0:
                break
            embedding = np.array(slide["embeddings"])
            if np.array(slide[label_tag]) == 1 and num_pos > 0 and embedding.shape[1] > 100 :
                num_pos -= 1
            elif np.array(slide[label_tag]) == 0 and num_neg > 0 and embedding.shape[1] > 100:
                num_neg -= 1
            else:
                continue
            embeddings.append(embedding[0,:,:])

    embeddings = np.vstack(embeddings)

    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    pca = PCA(n_components=0.99)
    results = pca.fit(scaled_embeddings)

    embeddings = pca.fit_transform(scaled_embeddings)
    kmeans = KMeans(n_clusters = NUM_CLUSTERS, init='k-means++', random_state = 0, n_init='auto')

    coords_dict = {}
    with h5py.File(embeddings_file, "r") as f:
        for slide in f.values():
            embedding = np.array(slide["embeddings"])[0,:,:]
            coords = np.array(slide["coords"])
            cluster_indexes = cluster_tiles(embedding, scaler, pca, kmeans)
            coords_indexes = []
            for cluster in range(NUM_CLUSTERS):
                indexes = np.where(cluster_indexes==cluster)[0]
                len_indexes = len(indexes)
                num_elements = NUM_ELEMENTS if NUM_ELEMENTS <= len_indexes else len_indexes
                selected_indexes = np.random.choice(indexes, size=num_elements, replace=False)
                coords_indexes.append(selected_indexes)
            
            coords_indexes = np.concatenate(coords_indexes)
            filtered_coords = coords[coords_indexes]
            
            coords_dict[str(slide["slide_id"][()].decode())] = np.array(filtered_coords)

    with open(coords_file, 'wb') as f:
        np.save(f, coords_dict)


def cluster_tiles(embeddings, scaler, pca, kmeans):
    scaled_embedding = scaler.fit_transform(embeddings)
    reduced_embedding = pca.fit_transform(scaled_embedding)
    cluster_indexes = kmeans.fit_predict(reduced_embedding)
    return cluster_indexes