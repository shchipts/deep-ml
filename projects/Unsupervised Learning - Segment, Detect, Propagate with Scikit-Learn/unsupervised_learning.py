import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_digits, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Step 1 - blobs_data
def blobs_data(random_state=42):
    centers = [[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8, 2.8], [-2.8, 1.3]]

    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1],
        random_state=random_state,
    )

    return X, y
    

# Step 2 - fit_kmeans
def fit_kmeans(X, k, random_state=42):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    kmeans.fit(X)
    return kmeans
    

# Step 3 - inertia_curve
def inertia_curve(X, ks):
    return dict([k, fit_kmeans(X, k).inertia_] for k in ks)
    

# Step 4 - silhouette_curve
def silhouette_curve(X, ks):
    def silhouette(k):
        nonlocal X

        kmeans = fit_kmeans(X, k)
        return silhouette_score(X, kmeans.predict(X))

    return dict([k, silhouette(k)] for k in ks)


def best_k_by_silhouette(curve):
    return sorted(curve.items(), key=lambda item: item[1], reverse=True)[0][0]


# Step 5 - fit_dbscan
def fit_dbscan(X, eps=0.2, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)
    return dbscan


def dbscan_summary(dbscan):
    labels = dbscan.labels_

    return {
        'n_clusters': int(len(set(labels) - {-1})),
        'n_noise': int(np.sum(labels == -1)),
        'n_core': int(len(dbscan.core_sample_indices_)),
    }


# Step 6 - dbscan_predict
def dbscan_predict(dbscan, X_new, n_neighbors=50):
    core_sample = dbscan.components_
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]

    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(core_sample, core_labels)
    return clf.predict(X_new)


# Step 7 - fit_gmm
def fit_gmm(X, n_components, random_state=42):
    return GaussianMixture(
        n_components=n_components, n_init=10, random_state=random_state
    ).fit(X)

def bic_curve(X, ks):
    return dict([k, fit_gmm(X, k).bic(X)] for k in ks)
    

# Step 8 - flag_anomalies
def flag_anomalies(gmm, X, contamination=0.04):
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)
    return np.array([int(x < threshold) for x in densities], dtype=bool)


# Step 9 - digits_data
def digits_data(test_size=0.25, random_state=42):
    X, y = load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
    

# Step 10 - baseline_50_random
def baseline_50_random(X_train, y_train, X_test, y_test, n_labeled=50, random_state=42):
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train[:n_labeled], y_train[:n_labeled])

    return float(accuracy_score(y_test, clf.predict(X_test)))


# Step 11 - representative_digits
def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state)
    distances = kmeans.transform(X_train)
    closest = np.array([np.argmin(distances[:, j]) for j in range(distances.shape[1])])
    return kmeans, closest


# Step 12 - train_on_representatives
def train_on_representatives(X_train, y_train, rep_idx, X_test, y_test):
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train[rep_idx], y_train[rep_idx])

    return float(accuracy_score(y_test, clf.predict(X_test)))
    

# Step 13 - propagate_and_train
def propagate_and_train(
    X_train, y_train, kmeans, rep_idx, X_test, y_test, percentile=20
):
    distances = kmeans.transform(X_train)
    cluster_labels = kmeans.labels_

    propagated_labels = np.empty(len(X_train), dtype=int)
    selected = np.zeros(len(X_train), dtype=bool)

    for j in range(kmeans.n_clusters):
        cluster_mask = cluster_labels == j
        cluster_distances = distances[cluster_mask, j]

        threshold = np.percentile(cluster_distances, percentile)

        selected[cluster_mask] = cluster_distances <= threshold
        propagated_labels[cluster_mask] = y_train[rep_idx[j]]

    X_selected = X_train[selected]
    y_selected = propagated_labels[selected]

    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_selected, y_selected)

    return {
        'n_propagated': int(np.sum(selected)),
        'label_accuracy': float(np.mean(y_selected == y_train[selected])),
        'test_accuracy': float(accuracy_score(y_test, clf.predict(X_test))),
    }


# Step 14 - synthetic_image
def synthetic_image(size=48):
    image = np.zeros((size, size, 3), dtype=np.float64)

    half = size // 2
    gradient = np.linspace(0.0, 1.0, half)

    image[:, :half, 0] = 1.0
    image[:, :half, 1] = gradient

    image[:, half:, 1] = np.linspace(0.0, 1.0, size - half)
    image[:, half:, 2] = 1.0

    square = size // 4
    image[:square, :square, :] = 1.0

    return image


# Step 15 - segment_colors
def segment_colors(image, k=4, random_state=42):
    rgb_colors = image.shape[2]
    pixels_as_list = image.reshape(-1, rgb_colors)
    kmeans_pixels = fit_kmeans(pixels_as_list, k, random_state)
    pixels_as_list_categorized = kmeans_pixels.cluster_centers_[kmeans_pixels.labels_]
    image_categorized = pixels_as_list_categorized.reshape(*image.shape)
    
    return image_categorized
    

# Step 16 - save_and_reload_clusterer
def save_and_reload_clusterer(kmeans, rep_labels, path):
    joblib.dump({'kmeans': kmeans, 'rep_labels': rep_labels}, path)
    image = joblib.load(path)

    return image


# Step 17 - predict_digit_labels
def predict_digit_labels(bundle, images):
    pixels_as_list = np.asarray(images).reshape(len(images), -1)
    pixels_categorized = bundle['kmeans'].predict(pixels_as_list)
    return bundle['rep_labels'][pixels_categorized].tolist()
