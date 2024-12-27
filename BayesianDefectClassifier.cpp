#include "BayesianDefectClassifier.h"

// Populates defect datasets and defect classes with statistical features
void BayesianDefectClassifier::fillDefectDatasetsAndDefectClasses(double distance, double angle)
{
    for (auto& defect : defects) {
        // Create a map of feature values for the defect
        std::map<std::string, double> stat_map;
        stat_map["mean"] = defect->getMean();
        stat_map["variance"] = defect->getVariance();
        stat_map["skewness"] = defect->getSkewness();
        stat_map["kurtosis"] = defect->getKurtosis();
        stat_map["inertia"] = defect->getInertia();
        stat_map["cluster_shade"] = defect->getClusterShade();
        stat_map["cluster_prominence"] = defect->getClusterProminence();
        stat_map["local_homogeneity"] = defect->getLocalHomogeneity();
        stat_map["energy"] = defect->getEnergy();
        stat_map["entropy"] = defect->getEntropy();

        // Check if the defect is a clear area
        if (defect->regionName == "Clear area") {
            clearWoodReference = stat_map;  // Set as reference for clear wood
            std::cout << "Clear Wood Reference: " << std::endl;
            for (const auto& entry : clearWoodReference) {
                std::cout << entry.first << ": " << entry.second << std::endl;
            }
        }
        else {
            defectDatasets.push_back(stat_map);
            defectClasses.push_back(defect->regionName);
        }
    }
}

// Forward sequential search to find the best feature subset
std::vector<std::string> BayesianDefectClassifier::forwardSequentialSearch(
    const std::map<std::string, double>& defect_i,
    const std::map<std::string, double>& defect_j)
{
    std::vector<std::string> selected_features;
    std::vector<std::string> remaining_features = featureSet;
    double best_likelihood_diff = -std::numeric_limits<double>::infinity();

    while (!remaining_features.empty()) {
        std::string best_feature;
        for (const auto& feature : remaining_features) {
            std::vector<std::string> test_features = selected_features;
            test_features.push_back(feature);

            std::vector<double> values_i, values_j;
            for (const auto& f : test_features) {
                values_i.push_back(defect_i.at(f));
                values_j.push_back(defect_j.at(f));
            }

            // Placeholder for actual likelihood calculation
            double likelihood_i = 0.0;
            double likelihood_j = 0.0;
            double likelihood_diff = std::abs(likelihood_i - likelihood_j);

            if (likelihood_diff > best_likelihood_diff) {
                best_likelihood_diff = likelihood_diff;
                best_feature = feature;
            }
        }

        if (!best_feature.empty()) {
            selected_features.push_back(best_feature);
            remaining_features.erase(
                std::remove(remaining_features.begin(), remaining_features.end(), best_feature),
                remaining_features.end());
        }
        else {
            break;
        }
    }

    return selected_features;
}

// Classifies the lumber defect based on the given region features
std::string BayesianDefectClassifier::classifyLumberDefect(const std::map<std::string, double>& regionFeatures)
{
    fillDefectDatasetsAndDefectClasses(0.0, 0.0);  // Placeholder for distance and angle
    int n = defectDatasets.size();
    std::vector<int> wins(n, 0);

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            auto selected_features = forwardSequentialSearch(defectDatasets[i], defectDatasets[j]);

            std::vector<double> data_i, data_j;
            for (const auto& feature : selected_features) {
                data_i.push_back(regionFeatures.at(feature));
                data_j.push_back(defectDatasets[j].at(feature));
            }

            // Placeholder for actual likelihood calculation
            double likelihood_i = 0.0;
            double likelihood_j = 0.0;

            if (likelihood_i > likelihood_j) {
                wins[i] += 1;
            }
            else {
                wins[j] += 1;
            }
        }
    }

    int max_index = std::distance(wins.begin(), std::max_element(wins.begin(), wins.end()));
    return defectClasses[max_index];
}

// Checks if the test region of interest (ROI) is a clear area based on the given threshold
bool BayesianDefectClassifier::isClearArea(const std::map<std::string, double>& testROI, double thresh)
{
    std::vector<double> x_vector, reference_vector;

    for (const auto& feature : clearWoodReference) {
        x_vector.push_back(testROI.at(feature.first));
        reference_vector.push_back(feature.second);
    }

    double test_statistic = 0.0;
    for (size_t i = 0; i < x_vector.size(); ++i) {
        test_statistic += std::pow(x_vector[i] - reference_vector[i], 2);
    }
    test_statistic /= x_vector.size();

    return test_statistic <= thresh;
}
