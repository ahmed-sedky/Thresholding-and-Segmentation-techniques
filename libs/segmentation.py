from black import out
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn import neighbors
from sqlalchemy import false
from libs import colorMap

def calculate_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def clusters_distance(cluster1, cluster2):
    return max([calculate_distance(point1, point2) for point1 in cluster1 for point2 in cluster2])

class KMeans:

    def __init__(self, K=5, max_iters=100):
        self.K = K
        self.max_iters = max_iters

        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def initialize_cluster_chracterstics(self,pixel_values):
        self.pixel_values = pixel_values
        self.n_samples, self.n_features = pixel_values.shape
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.pixel_values[idx] for idx in random_sample_idxs]

    def segment(self, pixel_values):
        self.initialize_cluster_chracterstics(pixel_values)

        for _ in range(self.max_iters):
            self.clusters = self.update_clusters(self.centroids)

            centroids_old = self.centroids
            self.centroids = self.update_centroids(self.clusters)

            if self.is_converged(centroids_old, self.centroids):
                break

        return self.get_cluster_labels(self.clusters)

    def get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def update_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.pixel_values):
            centroid_idx = self.closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    @staticmethod
    def closest_centroid(sample, centroids):
        distances = [calculate_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def update_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.pixel_values[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def is_converged(self, centroids_old, centroids):
        distances = [calculate_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def centroids_locations(self):
        return self.centroids

def apply_k_means(source, k=5, max_iter=100):
    b,g,r = cv2.split(source)       
    img = cv2.merge([r,g,b])     
    source = colorMap.RGB2LUV(img)
    pixel_values = source.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    model = KMeans(K=k, max_iters=max_iter)
    cluster_labels = model.segment(pixel_values)
    centertiods_loc = np.uint8(model.centroids_locations())
    cluster_labels = cluster_labels.astype(int)

    cluster_labels = cluster_labels.flatten()
    segmented_image = centertiods_loc[cluster_labels.flatten()]
    segmented_image = segmented_image.reshape(source.shape)

    return segmented_image

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


def get_around_pixels():
    around = [Point(-1, -1), Point(0, -1), Point(1, -1),
                Point(1, 0), Point(1, 1), Point(0, 1),
                Point(-1, 1), Point(-1, 0)]
    return around

def regionGrow(img, seeds, thresh):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)

    label = 1
    rest_8pixels_in_kernel = get_around_pixels()

    while (len(seeds) > 0):
        currentPoint = seeds.pop(0)

        seedMark[currentPoint.x, currentPoint.y] = label

        for i in range(8):
            neighbor_x = currentPoint.x + rest_8pixels_in_kernel[i].x
            neighbor_y = currentPoint.y + rest_8pixels_in_kernel[i].y

            if neighbor_x < 0 or neighbor_y < 0 or neighbor_x >= height or neighbor_y >= weight:
                continue

            grayDiff = getGrayDiff(img, currentPoint, Point(neighbor_x, neighbor_y))

            if grayDiff < thresh and seedMark[neighbor_x, neighbor_y] == 0:
                seedMark[neighbor_x, neighbor_y] = label
                seeds.append(Point(neighbor_x, neighbor_y))

    return seedMark

def select_random_seed(img,seeds):
    for i in range(3):
        x = np.random.randint(0, img.shape[0])
        y = np.random.randint(0, img.shape[1])
        seeds.append(Point(x, y))

def apply_region_growing(source: np.ndarray):

    src = np.copy(source)
    color_img = cv2.cvtColor(src, cv2.COLOR_Luv2BGR)
    img_gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    global seeds
    seeds = []   
    select_random_seed(img_gray,seeds)
    output_image = regionGrow(img_gray, seeds, 10)

    return output_image

# mean shift algorithm
def extract_features(image: np.ndarray):
    features = np.zeros((image.shape[0] * image.shape[1], 5))
    pixel_counter = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i][j]
            for k in range(5):
                if (k >= 0) & (k <= 2):
                    features[pixel_counter][k] = pixel[k]
                else:
                    if k == 3:
                        features[pixel_counter][k] = i
                    else:
                        features[pixel_counter][k] = j
            pixel_counter += 1
    return features

def get_thresholds(random_mean, threshold,features, current_mean):
    new_features_indices = []

    if random_mean:
        current_mean = np.random.randint(0, len(features))
        current_mean = features[current_mean]

    for feature_index, feature in enumerate(features):
        ecl_dist = calculate_distance(current_mean, feature)

        if ecl_dist < threshold:
            new_features_indices.append(feature_index)

    return new_features_indices, current_mean

def calculate_new_mean(features,new_features_indices, current_mean, segmented_image):
    iteration = 0.01
    red_mean = np.mean(features[new_features_indices][:, 0])
    green_mean = np.mean(features[new_features_indices][:, 1])
    blue_mean = np.mean(features[new_features_indices][:, 2])
    rows_mean = np.mean(features[new_features_indices][:, 3])
    columns_mean = np.mean(features[new_features_indices][:, 4])

    mean_e_distance = (calculate_distance(red_mean, current_mean[0]) +
                        calculate_distance(green_mean, current_mean[1]) +
                        calculate_distance(blue_mean, current_mean[2]) +
                        calculate_distance(rows_mean, current_mean[3]) +
                        calculate_distance(columns_mean, current_mean[4]))

    if mean_e_distance < iteration:
        rgb_pixel = np.zeros((1, 3))
        rgb_pixel[0][0] = red_mean
        rgb_pixel[0][1] = green_mean
        rgb_pixel[0][2] = blue_mean

        for i in range(len(new_features_indices)):
            m = int(features[new_features_indices[i]][3])
            n = int(features[new_features_indices[i]][4])
            segmented_image[m][n] = rgb_pixel

            features[new_features_indices[i]][0] = -1

        random_mean = True
        new_d = np.zeros((len(features), 5))
        itr = 0

        for i in range(len(features)):
            if features[i][0] != -1:
                for k in range(5):
                    new_d[itr][k] = features[i][k]
                itr += 1

        features = np.zeros((itr, 5))

        itr -= 1
        for i in range(itr):
            for k in range(5):
                features[i][k] = new_d[i][k]
            features[i][1] = new_d[i][1]
    else:
        random_mean = False
        current_mean = [red_mean, green_mean, blue_mean, rows_mean, columns_mean]
        
    return random_mean,features,current_mean,segmented_image   

def apply_mean_shift(image, threshold: int = 60):
    src = np.copy(image)
    random_mean = True
    current_mean = []

    size = src.shape[0], src.shape[1], 3
    segmented_image = np.zeros(size, dtype=np.uint8)

    features = extract_features(src)
    while len(features) > 0:
            new_features_indices, current_mean = get_thresholds(
                random_mean,
                threshold, features,  current_mean)

            random_mean,features,current_mean,segmented_image = calculate_new_mean(features,new_features_indices, current_mean, segmented_image)
    segmented_image = segmented_image

    return segmented_image

# agglomerative algorithm

clusters_list = []
cluster = {}
centers = {}

def clusters_average_distance(cluster1, cluster2):
   
    cluster1_center = np.average(cluster1)
    cluster2_center = np.average(cluster2)
    return calculate_distance(cluster1_center, cluster2_center) 

def initial_clusters(image_clusters):
  
    global initial_k
    groups = {}
    cluster_color = int(256 / initial_k)
    for i in range(initial_k):
        color = i * cluster_color
        groups[(color, color, color)] = []
    for i, p in enumerate(image_clusters):
        go = min(groups.keys(), key=lambda c: np.sqrt(np.sum((p - c) ** 2)))
        groups[go].append(p)
    return [group for group in groups.values() if len(group) > 0]

def get_cluster_center( point):
    global cluster
    point_cluster_num = cluster[tuple(point)]
    center = centers[point_cluster_num]
    return center

def get_clusters(image_clusters):
    global clusters_list
    clusters_list = initial_clusters(image_clusters)

    while len(clusters_list) > clusters_number:
        cluster1, cluster2 = min(
            [(c1, c2) for i, c1 in enumerate(clusters_list) for c2 in clusters_list[:i]],
            key=lambda c: clusters_average_distance(c[0], c[1]))

        clusters_list = [cluster_itr for cluster_itr in clusters_list if cluster_itr != cluster1 and cluster_itr != cluster2]

        merged_cluster = cluster1 + cluster2

        clusters_list.append(merged_cluster)

    global cluster 
    for cl_num, cl in enumerate(clusters_list):
        for point in cl:
            cluster[tuple(point)] = cl_num

    global centers 
    for cl_num, cl in enumerate(clusters_list):
        centers[cl_num] = np.average(cl, axis=0)

def apply_agglomerative_clustering(image, number_of_clusters,initial_number_of_clusters):
    global clusters_number
    global initial_k

    clusters_number = number_of_clusters
    initial_k = initial_number_of_clusters 
    flattened_image = np.copy(image.reshape((-1, 3)))

    get_clusters(flattened_image)
    output_image = []
    for row in image:
        rows = []
        for col in row:
            rows.append(get_cluster_center(list(col)))
        output_image.append(rows)    
    output_image = np.array(output_image, np.uint8)
    return output_image    