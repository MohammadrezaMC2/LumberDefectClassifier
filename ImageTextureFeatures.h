#ifndef IMAGETEXTUREFEATURES_H
#define IMAGETEXTUREFEATURES_H

#include <opencv2/core.hpp>  // OpenCV core module for Mat class
#include <tuple>  // To use std::tuple for storing moments
#include <string>

// Class for extracting texture features from an image section using the co-occurrence matrix and other statistical measures.
class ImageTextureFeatures {
private:
    cv::Mat imageSection;          // A section of the image (e.g., a patch or region of interest)
    cv::Mat cooccurrenceMatrix;    // Co-occurrence matrix for texture analysis
    double angle;                  // Angle used for co-occurrence matrix calculation
    double distance;               // Distance used for co-occurrence matrix calculation
    std::tuple<double, double> moments;  // Tuple to store the first and second moments

    // Statistical features
    double mean;
    double variance;
    double skewness;
    double kurtosis;

    // Texture features based on the co-occurrence matrix
    double clusterShade;
    double clusterProminence;
    double localHomogeneity;
    double energy;
    double entropy;
    double inertia;

    // Helper function to convert degrees to radians
    double degToRad(double degree);

public:
    std::string regionName;  // Name of the image region

    // Constructor: Initializes the image section, angle, and distance
    ImageTextureFeatures(const cv::Mat& Image, const std::string& RegionName, double Angle, double Distance);

    // Destructor
    virtual ~ImageTextureFeatures() {}

    // Getters for statistical and texture features
    double getMean() const;
    double getVariance() const;
    double getSkewness() const;
    double getKurtosis() const;

    double getClusterShade() const;
    double getClusterProminence() const;
    double getLocalHomogeneity() const;
    double getEnergy() const;
    double getEntropy() const;
    double getInertia() const;

    // Set the angle and distance for the co-occurrence matrix calculation
    void setAngleandDistance(double Angle, double Distance);

    // Methods to calculate the various features
    void calculateMean();                  // Calculates the mean of pixel values
    void calculateVariance();              // Calculates the variance of pixel values
    void calculateSkewness();              // Calculates the skewness of pixel values
    void calculateKurtosis();              // Calculates the kurtosis of pixel values
    void calculateCooccurrenceMatrix();    // Calculates the co-occurrence matrix for texture analysis
    void calculateMoments();               // Calculates the first and second moments
    void calculateClusterShade();          // Calculates the cluster shade of the co-occurrence matrix
    void calculateClusterProminence();     // Calculates the cluster prominence of the co-occurrence matrix
    void calculateLocalHomogeneity();      // Calculates the local homogeneity of the co-occurrence matrix
    void calculateEnergy();                // Calculates the energy of the co-occurrence matrix
    void calculateEntropy();               // Calculates the entropy of the co-occurrence matrix
    void calculateInertia();               // Calculates the inertia of the co-occurrence matrix

    // Calculate all texture features
    void calculateTextureFeatures();
};

#endif // IMAGETEXTUREFEATURES_H
