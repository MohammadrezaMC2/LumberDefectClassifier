import cv2
import numpy as np
import math

class ImageTextureFeatures:
    def __init__(self, image, region_name, angle, distance):
        self.image_section = image
        self.cooccurrence_matrix = np.zeros((256, 256), dtype=np.float64)  # Assuming 256 gray levels
        self.angle = angle
        self.distance = distance
        self.region_name = region_name

        # Initialize statistical and texture features
        self.mean = 0
        self.variance = 0
        self.skewness = 0
        self.kurtosis = 0
        self.cluster_shade = 0
        self.cluster_prominence = 0
        self.local_homogeneity = 0
        self.energy = 0
        self.entropy = 0
        self.inertia = 0

        # Verify if the image is grayscale
        if len(image.shape) != 2:
            raise ValueError("The input image is not a grayscale image.")

        # Perform calculations
        self.calculate_mean()
        self.calculate_variance()
        self.calculate_skewness()
        self.calculate_kurtosis()
        self.calculate_texture_features()

    def calculate_mean(self):
        self.mean = np.mean(self.image_section)

    def calculate_variance(self):
        self.variance = np.var(self.image_section)

    def calculate_skewness(self):
        mean_diff = self.image_section - self.mean
        self.skewness = np.mean(mean_diff**3) / (self.variance ** 1.5)

    def calculate_kurtosis(self):
        mean_diff = self.image_section - self.mean
        self.kurtosis = np.mean(mean_diff**4) / (self.variance ** 2)

    def calculate_cooccurrence_matrix(self):
        rows, cols = self.image_section.shape
        angle_rad = math.radians(self.angle)
        dx = int(round(self.distance * math.cos(angle_rad)))
        dy = int(round(self.distance * math.sin(angle_rad)))

        cooccurrence_freq = np.zeros((256, 256), dtype=np.int32)
        total_pairs = 0

        for i in range(rows):
            for j in range(cols):
                pixel_value = self.image_section[i, j]
                neighbor_x = i + dx
                neighbor_y = j + dy

                if 0 <= neighbor_x < rows and 0 <= neighbor_y < cols:
                    neighbor_value = self.image_section[neighbor_x, neighbor_y]
                    cooccurrence_freq[pixel_value, neighbor_value] += 1
                    total_pairs += 1

        if total_pairs > 0:
            self.cooccurrence_matrix = cooccurrence_freq / total_pairs

    def calculate_moments(self):
        indices = np.arange(256)
        Mx1 = np.sum(indices[:, None] * self.cooccurrence_matrix)
        Mx2 = np.sum(indices[None, :] * self.cooccurrence_matrix)
        return Mx1, Mx2

    def calculate_cluster_shade(self):
        Mx1, Mx2 = self.calculate_moments()
        indices = np.arange(256)
        diff = indices[:, None] + indices[None, :] - Mx1 - Mx2
        self.cluster_shade = np.sum((diff**3) * self.cooccurrence_matrix)

    def calculate_cluster_prominence(self):
        Mx1, Mx2 = self.calculate_moments()
        indices = np.arange(256)
        diff = indices[:, None] + indices[None, :] - Mx1 - Mx2
        self.cluster_prominence = np.sum((diff**4) * self.cooccurrence_matrix)

    def calculate_local_homogeneity(self):
        indices = np.arange(256)
        diff = indices[:, None] - indices[None, :]
        self.local_homogeneity = np.sum(self.cooccurrence_matrix / (1 + diff**2))

    def calculate_energy(self):
        self.energy = np.sum(self.cooccurrence_matrix**2)

    def calculate_entropy(self):
        with np.errstate(divide='ignore'):  # Suppress warnings for log(0)
            entropy_values = self.cooccurrence_matrix * np.log2(self.cooccurrence_matrix + 1e-12)
        self.entropy = -np.sum(entropy_values)

    def calculate_inertia(self):
        indices = np.arange(256)
        diff = (indices[:, None] - indices[None, :])**2
        self.inertia = np.sum(diff * self.cooccurrence_matrix)

    def calculate_texture_features(self):
        self.calculate_cooccurrence_matrix()
        self.calculate_cluster_shade()
        self.calculate_cluster_prominence()
        self.calculate_local_homogeneity()
        self.calculate_energy()
        self.calculate_entropy()
        self.calculate_inertia()

    def set_angle_and_distance(self, angle, distance):
        self.angle = angle
        self.distance = distance
        self.calculate_texture_features()

# Usage example for testing:
if __name__ == "__main__":
    image = cv2.imread("image_path", cv2.IMREAD_GRAYSCALE)
    region_name = "Test Region"
    angle = 45  # degrees
    distance = 1  # pixels

    features = ImageTextureFeatures(image, region_name, angle, distance)
    print("Mean:", features.mean)
    print("Variance:", features.variance)
