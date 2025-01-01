#include "ImageTextureFeatures.h"
#include <iostream>
#include <cmath>

// Constructor: Initializes the image section, angle, distance, and region name
ImageTextureFeatures::ImageTextureFeatures(const cv::Mat& Image, const std::string& RegionName, double Angle, double Distance)
    : imageSection{ Image },
    cooccurrenceMatrix{ cv::Mat::zeros(256, 256, CV_64FC1) },  // Assuming 256 gray levels for 8-bit image
    angle{ Angle },
    distance{ Distance },
    regionName{ RegionName } {

    // Check if the input image is grayscale
    if (imageSection.channels() != 1) {
        std::cerr << "The input image is not a grayscale image." << std::endl;
    }
    else {
        calculateMean();
        calculateVariance();
        calculateSkewness();
        calculateKurtosis();
        calculateTextureFeatures();
    }
}

// Getters for statistical and texture features
double ImageTextureFeatures::getMean() const { return mean; }
double ImageTextureFeatures::getVariance() const { return variance; }
double ImageTextureFeatures::getSkewness() const { return skewness; }
double ImageTextureFeatures::getKurtosis() const { return kurtosis; }
double ImageTextureFeatures::getClusterShade() const { return clusterShade; }
double ImageTextureFeatures::getClusterProminence() const { return clusterProminence; }
double ImageTextureFeatures::getLocalHomogeneity() const { return localHomogeneity; }
double ImageTextureFeatures::getEnergy() const { return energy; }
double ImageTextureFeatures::getEntropy() const { return entropy; }
double ImageTextureFeatures::getInertia() const { return inertia; }

// Sets the angle and distance for co-occurrence matrix calculation and recalculates texture features
void ImageTextureFeatures::setAngleandDistance(double Angle, double Distance) {
    angle = Angle;
    distance = Distance;
    calculateTextureFeatures();
}

// Calculates the mean of the pixel values in the image section
void ImageTextureFeatures::calculateMean() {
    cv::Scalar meanValue = cv::mean(imageSection);  // OpenCV's mean function computes the mean of each channel
    mean = meanValue[0];  // Since it's a grayscale image, meanValue[0] will hold the mean
}

// Calculates the variance of the pixel values
void ImageTextureFeatures::calculateVariance() {
    cv::Mat temp;
    cv::pow(imageSection - mean, 2, temp);  // Compute squared difference from mean
    cv::Scalar varianceValue = cv::mean(temp);  // Compute the mean of squared differences
    variance = varianceValue[0];  // Variance is the mean of squared differences
}

// Calculates the skewness of the pixel values
void ImageTextureFeatures::calculateSkewness() {
    cv::Mat temp;
    cv::pow(imageSection - mean, 3, temp);  // Compute cubed difference from mean
    cv::Scalar meanCubedDiff = cv::mean(temp);  // Compute the mean of cubed differences
    skewness = meanCubedDiff[0] / std::pow(variance, 1.5);  // Skewness formula
}

// Calculates the kurtosis of the pixel values
void ImageTextureFeatures::calculateKurtosis() {
    cv::Mat temp;
    cv::pow(imageSection - mean, 4, temp);  // Compute fourth power difference from mean
    cv::Scalar meanFourthDiff = cv::mean(temp);  // Compute the mean of fourth power differences
    kurtosis = meanFourthDiff[0] / (variance * variance);  // Kurtosis formula
}

// Calculates the co-occurrence matrix based on the given angle and distance
void ImageTextureFeatures::calculateCooccurrenceMatrix() {
    int rows = imageSection.rows;
    int cols = imageSection.cols;
    int num_levels = 256;  // Assuming 256 gray levels (8-bit grayscale)
    cv::Mat cooccurrence_freq = cv::Mat::zeros(num_levels, num_levels, CV_32S);

    float angle_rad = angle * CV_PI / 180.0f;  // Convert angle to radians
    int dx = static_cast<int>(round(distance * cos(angle_rad)));  // Calculate displacement in x
    int dy = static_cast<int>(round(distance * sin(angle_rad)));  // Calculate displacement in y

    int total_pairs = 0;  // Total number of valid pixel pairs

    // Loop through the image to fill the co-occurrence frequency matrix
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uchar pixel_value = imageSection.at<uchar>(i, j);  // Get the pixel value at (i, j)
            int neighbor_x = i + dx;  // Calculate the neighbor's x position
            int neighbor_y = j + dy;  // Calculate the neighbor's y position

            // Ensure the neighbor is within bounds
            if (neighbor_x >= 0 && neighbor_x < rows && neighbor_y >= 0 && neighbor_y < cols) {
                uchar neighbor_value = imageSection.at<uchar>(neighbor_x, neighbor_y);
                cooccurrence_freq.at<int>(pixel_value, neighbor_value) += 1;  // Increment the co-occurrence frequency
                total_pairs += 1;  // Increment total valid pairs
            }
        }
    }

    // Convert frequency matrix to a probability matrix
    if (total_pairs > 0) {
        cooccurrence_freq = cooccurrence_freq / total_pairs;
    }

    // Store the resulting co-occurrence matrix
    cooccurrenceMatrix = cooccurrence_freq;
}

// Calculates the first and second moments (Mx1 and Mx2) of the co-occurrence matrix
void ImageTextureFeatures::calculateMoments() {
    int num_levels = cooccurrenceMatrix.rows;
    cv::Mat indices = cv::Mat(num_levels, 1, CV_32S);
    for (int i = 0; i < num_levels; ++i) {
        indices.at<int>(i, 0) = i;  // Fill indices with pixel levels [0, num_levels)
    }

    double Mx1 = 0;
    double Mx2 = 0;

    // Calculate Mx1 (mean of pixel values)
    for (int i = 0; i < num_levels; ++i) {
        for (int j = 0; j < num_levels; ++j) {
            Mx1 += indices.at<int>(i, 0) * cooccurrenceMatrix.at<int>(i, j);
        }
    }

    // Calculate Mx2 (mean of target pixel values)
    for (int i = 0; i < num_levels; ++i) {
        for (int j = 0; j < num_levels; ++j) {
            Mx2 += indices.at<int>(j, 0) * cooccurrenceMatrix.at<int>(i, j);
        }
    }

    moments = std::make_tuple(Mx1, Mx2);  // Store the moments in a tuple
}

// Calculates the cluster shade of the co-occurrence matrix
void ImageTextureFeatures::calculateClusterShade() {
    double Mx1 = std::get<0>(moments);
    double Mx2 = std::get<1>(moments);
    int num_levels = cooccurrenceMatrix.rows;
    clusterShade = 0.0;

    // Calculate the cluster shade using the formula
    for (int i = 0; i < num_levels; ++i) {
        for (int j = 0; j < num_levels; ++j) {
            double diff = i + j - Mx1 - Mx2;
            clusterShade += std::pow(diff, 3) * cooccurrenceMatrix.at<int>(i, j);
        }
    }
}

// Calculates the cluster prominence of the co-occurrence matrix
void ImageTextureFeatures::calculateClusterProminence() {
    double Mx1 = std::get<0>(moments);
    double Mx2 = std::get<1>(moments);
    int num_levels = cooccurrenceMatrix.rows;
    clusterProminence = 0.0;

    // Calculate the cluster prominence using the formula
    for (int i = 0; i < num_levels; ++i) {
        for (int j = 0; j < num_levels; ++j) {
            double diff = i + j - Mx1 - Mx2;
            clusterProminence += std::pow(diff, 4) * cooccurrenceMatrix.at<int>(i, j);
        }
    }
}

// Calculates the local homogeneity of the co-occurrence matrix
void ImageTextureFeatures::calculateLocalHomogeneity() {
    int num_levels = cooccurrenceMatrix.rows;
    localHomogeneity = 0.0;

    // Calculate the local homogeneity using the formula
    for (int i = 0; i < num_levels; ++i) {
        for (int j = 0; j < num_levels; ++j) {
            double homogeneity_term = 1.0 / (1 + std::pow(i - j, 2));  // Homogeneity term formula
            localHomogeneity += homogeneity_term * cooccurrenceMatrix.at<int>(i, j);
        }
    }
}

// Calculates the energy of the co-occurrence matrix
void ImageTextureFeatures::calculateEnergy() {
    energy = 0.0;
    int num_levels = cooccurrenceMatrix.rows;

    // Calculate the energy using the formula
    for (int i = 0; i < num_levels; ++i) {
        for (int j = 0; j < num_levels; ++j) {
            energy += std::pow(cooccurrenceMatrix.at<int>(i, j), 2);
        }
    }
}

// Calculates the entropy of the co-occurrence matrix
void ImageTextureFeatures::calculateEntropy() {
    entropy = 0.0;
    int num_levels = cooccurrenceMatrix.rows;

    // Calculate the entropy using the formula
    for (int i = 0; i < num_levels; ++i) {
        for (int j = 0; j < num_levels; ++j) {
            double p = cooccurrenceMatrix.at<int>(i, j);  // Probability value
            if (p > 0) {
                entropy -= p * std::log2(p);  // Entropy formula
            }
        }
    }
}

// Calculates the inertia of the co-occurrence matrix
void ImageTextureFeatures::calculateInertia() {
    inertia = 0.0;
    int num_levels = cooccurrenceMatrix.rows;

    // Calculate the inertia using the formula
    for (int i = 0; i < num_levels; ++i) {
        for (int j = 0; j < num_levels; ++j) {
            inertia += std::pow(i - j, 2) * cooccurrenceMatrix.at<int>(i, j);
        }
    }
}

// Calculates all the texture features by calling respective functions
void ImageTextureFeatures::calculateTextureFeatures() {
    calculateCooccurrenceMatrix();
    calculateMoments();
    calculateClusterShade();
    calculateClusterProminence();
    calculateLocalHomogeneity();
    calculateEnergy();
    calculateEntropy();
    calculateInertia();
}
