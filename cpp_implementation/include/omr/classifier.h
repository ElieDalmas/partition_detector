#pragma once

#include "omr/dataset.h"
#include "omr/knn.h"

#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace omr {
    // level 1 categories
    extern const std::unordered_set<std::string> NOTEHEAD_CATS;
    extern const std::unordered_set<std::string> REST_CATS;
    extern const std::unordered_set<std::string> CLEF_CATS;

    // level 2 notehead subtypes
    extern const std::unordered_set<std::string> NOTEHEAD_BLACK;
    extern const std::unordered_set<std::string> NOTEHEAD_HALF;
    extern const std::unordered_set<std::string> NOTEHEAD_WHOLE;

    std::string level1Label(const std::string &catName);

    std::optional<std::string> level2Label(const std::string &catName);

    struct TrainingData {
        std::vector<std::vector<double> > X;
        std::vector<std::string> yL1;
        std::vector<std::optional<std::string> > yL2; // std::nullopt if not notehead
    };

    TrainingData buildTrainingData(const std::vector<Annotation> &anns,
                                   const std::unordered_map<std::string, std::string> &idToName,
                                   const cv::Mat &imgBin);

    // Two-level kNN classifier
    // Level 1: notehead, rest, clef, other
    // Level 2: black, half, whole (noteheads only)
    class SymbolClassifier {
    public:
        explicit SymbolClassifier(std::size_t k = 5);

        void fit(const std::vector<std::vector<double> > &X, const std::vector<std::string> &yL1,
                 const std::vector<std::optional<std::string> > &yL2);

        // Return (level1, level2) pairs
        std::vector<std::pair<std::string, std::optional<std::string> > > predict(
            const std::vector<std::vector<double> > &X) const;

    private:
        KNeighborsClassifier _knnL1;
        KNeighborsClassifier _knnL2;
    };
}
