import imp
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
import time

from black import out

import algorithms.color_map as ColorMap
import algorithms.segmentation as Segmentation
import algorithms.threshold as Threshold
import modules.message as Message
import modules.image as Image

output_image_path = "cached/output.png"


def browse_files(self, input_image):
    file_name = QFileDialog.getOpenFileName(
        self, "Open file", "./test", "*.jpg;;" " *.png;;" "*.jpeg;;"
    )
    file_path = file_name[0]

    extensionsToCheck = (".jpg", ".png", ".jpeg", ".jfif")
    if file_name[0].endswith(extensionsToCheck):
        start(self, file_path, input_image)
    elif file_name[0] != "":
        Message.error(self, "Invalid format.")
        return
    else:
        return


def start(self, file_path, input_image):
    global original_image_matrix

    self.output_image.clear()
    plot_image(self, file_path, input_image)
    enable_actions(self)

    if input_image == "original":
        original_image_matrix = Image.read(file_path)


def plot_image(self, image_path, image_type):
    if image_type == "original":
        self.original_image.setPhoto(QPixmap(image_path))
    if image_type == "output":
        self.output_image.setPhoto(QPixmap(image_path))


def enable_actions(self):
    self.features_combobox.setEnabled(True)


def choose_feature(self, text):
    global feature

    feature = text
    start = time.time()
    if text == "RGB to LUV":
        output_image = ColorMap.RGB_to_LUV(original_image_matrix)
    elif text == "K-Means":
        output_image, _ = Segmentation.k_means(original_image_matrix)
    elif text == "Region Growing":
        output_image = Segmentation.region_growing(original_image_matrix)
    elif text == "Agglomerative Clustering":
        output_image = Segmentation.agglomerative_clustering(original_image_matrix)
    elif text == "Mean Shift":
        output_image = Segmentation.mean_shift(original_image_matrix)
    elif text == "Global Otsu":
        output_image = Threshold.global_otsu(original_image_matrix)
    elif text == "Global Optimal":
        output_image = Threshold.global_optimal(original_image_matrix)
    elif text == "Global Spectral":
        output_image = Threshold.global_spectral(original_image_matrix)
    elif text == "Local Otsu":
        output_image = Threshold.local_otsu(original_image_matrix)
    elif text == "Local Optimal":
        output_image = Threshold.local_optimal(original_image_matrix)
    elif text == "Local Spectral":
        output_image = Threshold.local_spectral(original_image_matrix)
    end = time.time()
    Image.write(output_image_path, output_image)
    plot_image(self, output_image_path, "output")
    Message.info(self, f"Time taken equals {round(end - start, 2)} seconds")
