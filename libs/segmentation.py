import numpy as np
import matplotlib.pyplot as plt
import cv2
from libs import colorMap

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

# KMeans Algorithm
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def clusters_distance(cluster1, cluster2):
    return max([euclidean_distance(point1, point2) for point1 in cluster1 for point2 in cluster2])

class KMeans:

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()

            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    @staticmethod
    def _closest_centroid(sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        plt.show()

    def cent(self):
        return self.centroids

def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1),
                    Point(1, 0), Point(1, 1), Point(0, 1),
                    Point(-1, 1), Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]

    return connects

def regionGrow(img, seeds, thresh, p = 1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []

    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)

    while (len(seedList) > 0):
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.x, currentPoint.y] = label

        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y

            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue

            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))

            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))

    return seedMark



def apply_k_means(source, k=5, max_iter=100):
    b,g,r = cv2.split(source)       
    img = cv2.merge([r,g,b])     
    source = colorMap.RGB2LUV(img)
    pixel_values = source.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    model = KMeans(K=k, max_iters=max_iter)
    y_pred = model.predict(pixel_values)

    centers = np.uint8(model.cent())
    y_pred = y_pred.astype(int)

    labels = y_pred.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(source.shape)

    return segmented_image, labels

def apply_region_growing(source: np.ndarray):

    src = np.copy(source)
    color_img = cv2.cvtColor(src, cv2.COLOR_Luv2BGR)
    img_gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    seeds = []
    for i in range(3):
        x = np.random.randint(0, img_gray.shape[0])
        y = np.random.randint(0, img_gray.shape[1])
        seeds.append(Point(x, y))

    output_image = regionGrow(img_gray, seeds, 10)

    return output_image

class MeanShift:
    def __init__(self, image, threshold):
        self.threshold = threshold
        self.apply_random_mean = True
        self.current_mean = []

        size = image.shape[0], image.shape[1], 3
        self.segmented_image = np.zeros(size, dtype=np.uint8)

        self.features = self.create_features(image=image)

    def mean_shift(self):
        while len(self.features) > 0:
            below_threshold, self.current_mean = self.euclidean_distance(
                apply_random_mean=self.apply_random_mean,
                threshold=self.threshold)

            self.calculate_new_mean(below_threshold=below_threshold)

    def get_segmented_image(self):
        return self.segmented_image

    @staticmethod
    def create_features(image: np.ndarray):
        features = np.zeros((image.shape[0] ,image.shape[1]))

        for i in range(0,image.shape[0]):
            for j in range(0,image.shape[1]):
                features[i][j] = ((int(image[i,j,0]) + int(image[i,j,1]) + int(image[i,j,2]))/3)
            return features

    def euclidean_distance(self, apply_random_mean, threshold):
        below_threshold = []

        if apply_random_mean:
            current_mean = np.random.randint(0, len(self.features))
            self.current_mean = self.features[current_mean]

        for f_indx, feature in enumerate(self.features):
            ecl_dist = self.euclidean_distance(self.current_mean, feature)

            if ecl_dist < threshold:
                below_threshold.append(f_indx)

        return below_threshold, self.current_mean

    def calculate_new_mean(self, below_threshold):
        iteration = 0.01
        mean_r = np.mean(self.features[below_threshold][:, 0])
        mean_g = np.mean(self.features[below_threshold][:, 1])
        mean_b = np.mean(self.features[below_threshold][:, 2])
        mean_i = np.mean(self.features[below_threshold][:, 3])
        mean_j = np.mean(self.features[below_threshold][:, 4])

        mean_e_distance = (self.euclidean_distance(mean_r, self.current_mean[0]) +
                           self.euclidean_distance(mean_g, self.current_mean[1]) +
                           self.euclidean_distance(mean_b, self.current_mean[2]) +
                           self.euclidean_distance(mean_i, self.current_mean[3]) +
                           self.euclidean_distance(mean_j, self.current_mean[4]))

        if mean_e_distance < iteration:
            new_arr = np.zeros((1, 3))
            new_arr[0][0] = mean_r
            new_arr[0][1] = mean_g
            new_arr[0][2] = mean_b

            for i in range(len(below_threshold)):
                m = int(self.features[below_threshold[i]][3])
                n = int(self.features[below_threshold[i]][4])
                self.output_array[m][n] = new_arr

                self.features[below_threshold[i]][0] = -1

            self.apply_random_mean = True
            new_d = np.zeros((len(self.features), 5))
            itr = 0

            for i in range(len(self.features)):
                if self.features[i][0] != -1:
                    new_d[itr][0] = self.features[i][0]
                    new_d[itr][1] = self.features[i][1]
                    new_d[itr][2] = self.features[i][2]
                    new_d[itr][3] = self.features[i][3]
                    new_d[itr][4] = self.features[i][4]
                    itr += 1

            self.features = np.zeros((itr, 5))

            itr -= 1
            for i in range(itr):
                self.features[i][0] = new_d[i][0]
                self.features[i][1] = new_d[i][1]
                self.features[i][2] = new_d[i][2]
                self.features[i][3] = new_d[i][3]
                self.features[i][4] = new_d[i][4]

        else:
            self.apply_random_mean = False
            self.current_mean[0] = mean_r
            self.current_mean[1] = mean_g
            self.current_mean[2] = mean_b
            self.current_mean[3] = mean_i
            self.current_mean[4] = mean_j

def apply_mean_shift(image, threshold: int = 60):
    src = np.copy(image)
    mean_shift = MeanShift(image=src, threshold=threshold)
    mean_shift.mean_shift()
    segmented_image = mean_shift.get_segmented_image()

    return segmented_image