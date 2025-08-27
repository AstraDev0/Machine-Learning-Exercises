import matplotlib.pyplot as plt
import random

def load_txt(file):
    with open(file, "r") as f:
        values = f.read().splitlines()
    return [[float(num) for num in line.split()] for line in values if line.strip()]

def sqrt(x, epsilon=1e-10):
    if x < 0:
        raise ValueError("Negative number")
    guess = x
    while abs(guess**2 - x) > epsilon:
        guess = (guess + x / guess) / 2
    return guess

def euclidean_distance(x_point, y_point):
    return sqrt(sum((x_i - y_i)**2 for x_i, y_i in zip(x_point, y_point)))

def compute_distortion(points, center):
    return sum(euclidean_distance(point, center)**2 for point in points)

def allclose_manual(a, b, atol=1e-9, rtol=1e-5):
    if len(b) != len(a) and len(b[0]) != len(a[0]):
        return False
    flatten = lambda values_list: [item for sublist in values_list for item in sublist]
    for a_elem, b_elem in zip(flatten(a), flatten(b)):
        if abs(a_elem - b_elem) > atol + rtol * abs(b_elem):
            return False
    return True

def mean(points):
    return [sum(x) / len(points) for x in zip(*points)]

def kmeans(centroids, dataset, max_iters=2):
    k = len(centroids)
    for iteration in range(max_iters):
        clusters = {i: [] for i in range(k)}
        for point in dataset:
            distances = [euclidean_distance(c, point) for c in centroids]
            clusters[distances.index(min(distances))].append(point)
        new_centroids = [mean(clusters[d]) if clusters[d] else centroids[d] for d in range(k)]
        plot_clusters(list(clusters.values()), centroids, f"K-means - Iteration {iteration + 1}")
        if allclose_manual(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids

def split_class(points, center):
    v = [random.uniform(-0.01, 0.01) for _ in range(len(center))]
    center_plus = [center[i] + v[i] for i in range(len(center))]
    center_minus = [center[i] - v[i] for i in range(len(center))]
    cluster_a, cluster_b = [], []
    for p in points:
        if euclidean_distance(p, center_plus) < euclidean_distance(p, center_minus):
            cluster_a.append(p)
        else:
            cluster_b.append(p)
    return cluster_a, cluster_b

def plot_clusters(clusters, centroids, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['blue', 'black', 'red', 'green', 'magenta']
    for i, cluster in enumerate(clusters):
        if cluster:
            x = [p[0] for p in cluster]
            y = [p[1] for p in cluster]
            z = [0] * len(cluster)
            ax.scatter(x, y, z, color=colors[i % len(colors)], label=f'Cluster {i}')
    for i, center in enumerate(centroids):
        ax.scatter(center[0], center[1], 0, color=colors[i % len(colors)], marker='X', s=100, edgecolor='k', label=f'Center {i}')
    ax.set_title(title)
    ax.legend()
    plt.show()

def run():
    dataset = [(2, 3),(7, 0.5),(11, 1),(4, -0.5),(10, 2.5),(3, 2),(7, 0),(9, 1.5),(4, 2.5),(10, 1)]
    initial_centroids = [(2, 3),(4, -0.5),(10, 2.5)]
    kmeans(initial_centroids, dataset)

if __name__ == "__main__":
    run()
