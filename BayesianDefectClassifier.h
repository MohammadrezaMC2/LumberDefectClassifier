#ifndef BAYESIANDEFECTCLASSIFIER_H
#define BAYESIANDEFECTCLASSIFIER_H

#include "ImageTextureFeatures.h"
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <memory>
#include <cmath>

// Class definition for BayesianDefectClassifier
class BayesianDefectClassifier
{
public:
    // Vector of shared pointers to ImageTextureFeatures objects representing defects
    std::vector<std::shared_ptr<ImageTextureFeatures>> defects;

    // Map representing the reference features for clear wood
    std::map<std::string, double> clearWoodReference;

    // Vector of maps, each representing the features of different defect datasets
    std::vector<std::map<std::string, double>> defectDatasets;

    // Vector of strings representing different defect classes
    std::vector<std::string> defectClasses;

    // Vector of strings representing the feature set used for classification
    std::vector<std::string> featureSet;

    // Constructor: Initializes the feature set
    BayesianDefectClassifier(const std::vector<std::string>& FeatureSet)
        : featureSet{ FeatureSet } {
    }

    // Fills the defect datasets and classes based on the given distance and angle
    void fillDefectDatasetsAndDefectClasses(double distance, double angle);

    // Performs forward sequential search to find the best feature subset
    std::vector<std::string> forwardSequentialSearch(
        const std::map<std::string, double>& defect_i,
        const std::map<std::string, double>& defect_j);

    // Classifies the lumber defect based on the given region features
    std::string classifyLumberDefect(const std::map<std::string, double>& regionFeatures);

    // Checks if the test region of interest (ROI) is a clear area based on the given threshold
    bool isClearArea(const std::map<std::string, double>& testROI, double thresh);
};

#endif // BAYESIANDEFECTCLASSIFIER_H
