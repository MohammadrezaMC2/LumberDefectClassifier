#include "BayesianDefectClassifier.h"
#include "ImageTextureFeatures.h"

int main() {
    // Define the feature set
    /*
    std::vector<std::string> feature_set = { "mean", "variance", "skewness", "kurtosis", "inertia", "cluster_shade", "cluster_prominence", "local_homogeneity", "energy", "entropy" };

    // Create a DefectDetection object
    BayesianDefectClassifier classifier(feature_set);

    // Create some defects and add them to the classifier
    auto noDefect = std::make_shared<ImageTextureFeatures>("Clear area");
    noDefect-> imageSection = cv2.imread("section_of_the_image_where_there_is_no_defect", cv2.IMREAD_GRAYSCALE);

    auto defect1 = std::make_shared<ImageTextureFeatures>("Defect A");
    defect1-> imageSection = cv2.imread("the section of the image where defect1 is present", cv2.IMREAD_GRAYSCALE)

    auto defect2 = std::make_shared<ImageTextureFeatures>("Defect B");
    defect2-> imageSection = cv2.imread("the section of the image where defect2 is present", cv2.IMREAD_GRAYSCALE)
    .
    .
    .

    classifier.defects.push_back(defect1);
    classifier.defects.push_back(defect2);

    // Classify defects
    std::map<std::string, double> region_features = { {"mean", 7.0}, {"variance", 2.5} };
    std::string result = classifier.classifyLumberDefect(region_features);

    std::cout << "Classified as: " << result << std::endl;

    return 0;
    */
}